import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import cv2
from model import UNet, RetinaDataset, dice_coefficient, iou_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt

def visualize_predictions(data, target, pred):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    # Select the first image in the batch for visualization
    data = data[0, 0].cpu() if data.dim() == 4 else data.cpu()
    target = target[0, 0].cpu() if target.dim() == 4 else target.cpu()
    pred = pred[0, 0].cpu() if pred.dim() == 4 else pred.cpu()
    
    ax[0].imshow(data, cmap='gray')
    ax[0].set_title('Input Image')
    
    ax[1].imshow(target, cmap='gray')
    ax[1].set_title('Ground Truth')
    
    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title('Prediction')
    
    plt.show()

def load_model(checkpoint_path, device):
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if the checkpoint is a dictionary with additional info
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model with additional training info.")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights only.")

    model.eval()  # Set model to evaluation mode
    return model

def test_model(model, loader, device):
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = torch.sigmoid(output) > 0.5  # Threshold predictions for binary segmentation
            dice = dice_coefficient(target, pred)
            iou = iou_score(target, pred)
            dice_scores.append(dice)
            iou_scores.append(iou)
        # visualize_predictions(data, target, pred)
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    return mean_dice, mean_iou

def main(checkpoint_path, test_images_path, test_masks_path, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define any required transformations (without normalization if you prefer not to)
    transform = A.Compose([
        A.Rotate(limit=45, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2()
    ])

    test_images = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path)]
    test_masks = [os.path.join(test_masks_path, f) for f in os.listdir(test_masks_path)]

    test_dataset = RetinaDataset(test_images, test_masks, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(checkpoint_path, device)
    mean_dice, mean_iou = test_model(model, test_loader, device)
    print(f"Test Dice Coefficient: {mean_dice:.4f}, Test IoU Score: {mean_iou:.4f}")

if __name__ == "__main__":
    checkpoint_path = "checkpoints/regularized_unet_epoch_20.pth"  # Update with your actual model checkpoint path
    test_images_path = "/Users/danielvennemeyer/Workspace/Deep Learning/Retinal-Segmentation/Data/test/image"
    test_masks_path = "/Users/danielvennemeyer/Workspace/Deep Learning/Retinal-Segmentation/Data/test/mask"
    main(checkpoint_path, test_images_path, test_masks_path)