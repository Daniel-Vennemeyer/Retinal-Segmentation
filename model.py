import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import device
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt


class RetinaDataset(Dataset):
    def __init__(self, image_paths, mask_paths, mask_transform=None, image_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    # Apply the transformations separately
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # Apply image transform
        image = self.image_transform(image)

        # Apply mask transform and convert to single channel if needed
        mask = self.mask_transform(mask)
        mask = mask.squeeze(0)  # Ensure mask has a single channel

        return image, mask

# Define data paths
train_image_dir = "/Users/danielvennemeyer/Workspace/Deep Learning/HW3/Data/train/image"
train_mask_dir = "/Users/danielvennemeyer/Workspace/Deep Learning/HW3/Data/train/mask"
train_images = [os.path.join(train_image_dir, img) for img in os.listdir(train_image_dir)]
train_masks= [os.path.join(train_mask_dir, mask) for mask in os.listdir(train_mask_dir)]

test_image_dir = "/Users/danielvennemeyer/Workspace/Deep Learning/HW3/Data/test/image"
test_mask_dir = "/Users/danielvennemeyer/Workspace/Deep Learning/HW3/Data/test/mask"
val_images = [os.path.join(test_image_dir, img) for img in os.listdir(test_image_dir)]
val_masks = [os.path.join(test_mask_dir, mask) for mask in os.listdir(test_mask_dir)]

# Split dataset
# train_images, val_images, train_masks, val_masks = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)


# Transformations and DataLoader
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])

from torchvision import transforms

# Image transformations (for 3-channel RGB images)
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize for 3 channels
])

# Mask transformations (for 1-channel binary masks)
mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])



train_dataset = RetinaDataset(train_images, train_masks, image_transform=image_transform, mask_transform=mask_transform)
val_dataset = RetinaDataset(val_images, val_masks, image_transform=image_transform, mask_transform=mask_transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)




class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Decoder
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, out_channels)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        
        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc2))
        
        # Decoder
        dec2 = self.dec2(torch.cat([nn.Upsample(scale_factor=2)(bottleneck), enc2], dim=1))
        dec1 = self.dec1(torch.cat([nn.Upsample(scale_factor=2)(dec2), enc1], dim=1))

        return torch.sigmoid(dec1)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)


def dice_loss(pred, target, smooth=1e-6):
    # Ensure the inputs are 4D: [batch_size, channels, height, width]
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)  # Add channel dimension if missing
    if target.dim() == 3:
        target = target.unsqueeze(1)  # Add channel dimension if missing

    # Compute Dice coefficient
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return 1 - dice.mean()


def combined_loss(pred, target, dice_weight=0.5, bce_weight=0.5):
    # Ensure target has the same shape as pred by adding a channel dimension if needed
    if target.dim() == 3:
        target = target.unsqueeze(1)  # Make target shape [batch_size, 1, height, width]

    dice = dice_loss(pred, target)
    bce = nn.BCELoss()(pred, target)
    return dice_weight * dice + bce_weight * bce

def calculate_metrics(pred, target, smooth=1e-6):
    # Ensure target has the same shape as pred by adding a channel dimension if needed
    if target.dim() == 3:
        target = target.unsqueeze(1)  # Make target shape [batch_size, 1, height, width]
    
    # Flatten the tensors along the spatial dimensions for element-wise comparison
    pred_flat = pred.view(pred.size(0), -1)  # Shape: [batch_size, height * width]
    target_flat = target.view(target.size(0), -1)  # Shape: [batch_size, height * width]

    # Calculate Intersection
    intersection = (pred_flat * target_flat).sum(dim=1)
    
    # Dice Score
    dice = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    mean_dice = dice.mean().item()  # Get the average Dice score across the batch

    # IoU Score
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    mean_iou = iou.mean().item()  # Get the average IoU score across the batch
    
    print(f"Dice={mean_dice:.4f}, IoU={mean_iou:.4f}")
    return mean_dice, mean_iou

# criterion = dice_loss
criterion = combined_loss
optimizer = optim.Adam(model.parameters(), lr=0.00001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = combined_loss(outputs, masks)  # Use combined BCE + Dice loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)

    # Adjust learning rate with the scheduler
    scheduler.step(train_loss)
    
    calculate_metrics(outputs, masks)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss / len(train_loader.dataset):.4f}")



model.eval()
dice_scores, iou_scores = [], []
with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        dice, iou = calculate_metrics(outputs, masks)
        dice_scores.append(dice)
        iou_scores.append(iou)

print(f"Mean Dice: {np.mean(dice_scores):.4f}, Mean IoU: {np.mean(iou_scores):.4f}")


import matplotlib.pyplot as plt

def show_predictions(model, val_loader):
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            # Get model predictions
            preds = model(images)
            
            # Select the first prediction and mask
            pred = preds[0].squeeze().cpu().numpy()  # Shape should now be (256, 256)
            mask = masks[0].squeeze().cpu().numpy()  # Shape should now be (256, 256)
            
            # Ensure values are in range [0, 1] for display
            pred = (pred + 1) / 2.0  # If pred was in range [-1, 1], rescale to [0, 1]
            mask = (mask + 1) / 2.0  # If mask was in range [-1, 1], rescale to [0, 1]

            # Plot the results
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(pred, cmap='gray')
            ax[0].set_title("Prediction")
            ax[1].imshow(mask, cmap='gray')
            ax[1].set_title("Ground Truth Mask")
            plt.show()
            
            # Stop after showing the first batch for simplicity
            break


show_predictions(model, val_loader)
print()
