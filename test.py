import torch, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics import jaccard_score

# Define U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Encoder (Downsampling)
        for feature in features:
            self.encoder.append(self.conv_block(in_channels, feature))
            in_channels = feature

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.decoder.append(self.conv_block(feature*2, feature))

        # Bottleneck
        self.bottleneck = self.conv_block(features[-1], features[-1]*2)

        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](x)
            
        return self.final_conv(x)

def dice_coefficient(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).float()  # Threshold predictions
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    dice = (2. * intersection + 1e-6) / (y_true_f.sum() + y_pred_f.sum() + 1e-6)
    return dice.item()

def iou_score(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).float()  # Threshold predictions
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()



# Training Loop
def train_model(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        target = (target > 0).float()  # Convert any non-zero values to 1
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

# Testing Model
def test_model(model, loader, device):
    model.eval()
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = torch.sigmoid(output) > 0.5
            dice = dice_coefficient(target, pred)
            iou = iou_score(target, pred)
            dice_scores.append(dice)
            iou_scores.append(iou)
    # visualize_predictions(data, target, pred)
    return np.mean(dice_scores), np.mean(iou_scores)

# Dataset Class
class RetinaDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true, smooth=1e-6):
        y_pred = torch.sigmoid(y_pred)  # Apply sigmoid for probabilities
        y_pred_f = y_pred.view(-1)
        y_true_f = y_true.view(-1)
        intersection = (y_pred_f * y_true_f).sum()
        return 1 - (2. * intersection + smooth) / (y_pred_f.sum() + y_true_f.sum() + smooth)

def combined_loss(y_pred, y_true, bce_weight=0.5):
    bce = nn.BCEWithLogitsLoss()(y_pred, y_true)
    dice = DiceLoss()(y_pred, y_true)
    return bce_weight * bce + (1 - bce_weight) * dice

import matplotlib.pyplot as plt

def visualize_predictions(data, target, pred):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(data[0, 0].cpu(), cmap='gray')
    ax[0].set_title('Input Image')
    ax[1].imshow(target[0, 0].cpu(), cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[2].imshow(pred[0, 0].cpu(), cmap='gray')
    # ax[2].set_title(f'Prediction Epoch {epoch}')
    plt.show()



# Loading Dataset and Training Model
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    train_images_path = '/Users/danielvennemeyer/Workspace/Deep Learning/HW3/Data/test/image'
    train_images = [os.path.join(train_images_path, f) for f in os.listdir(train_images_path)]
    train_mask_path = '/Users/danielvennemeyer/Workspace/Deep Learning/HW3/Data/test/mask'
    train_masks = [os.path.join(train_mask_path, f) for f in os.listdir(train_mask_path)]

    dataset = RetinaDataset(train_images, train_masks, transform=transform)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = UNet(in_channels=1, out_channels=1).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    criterion = combined_loss

    for epoch in range(400):  # Adjust epochs as needed
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        dice, iou = test_model(model, val_loader, device)
        scheduler.step(train_loss)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Dice={dice:.4f}, IoU={iou:.4f}")

if __name__ == "__main__":
    main()
