# Retinal Blood Vessel Segmentation with U-Net

This project aims to segment retinal blood vessels in medical images using a U-Net architecture. The segmentation task is crucial for diagnosing retinal diseases, such as diabetic retinopathy and macular degeneration. The U-Net architecture was chosen for its effectiveness in medical image segmentation, thanks to its encoder-decoder structure that captures both local and global context.

## Project Overview

- **Model**: U-Net with customized layers.
- **Framework**: PyTorch.
- **Transformations**: Extensive data augmentation using Albumentations.
- **Normalization**: Optional image normalization to enhance model generalization.

## Architectural Decisions

### 1. **U-Net Architecture Design**

The U-Net model is widely used for image segmentation due to its symmetrical encoder-decoder structure. Here are the key decisions made in designing the U-Net for this project:

- **Encoder Depth and Filters**: The encoder consists of three layers, with each layer increasing the number of filters. This configuration was chosen to capture complex features without overwhelming the dataset's capacity.
- **Skip Connections**: The model uses skip connections between the encoder and decoder layers. These connections help retain high-resolution features from early layers, improving segmentation accuracy, especially in boundary regions.
- **Bottleneck Layer**: A bottleneck was added between the encoder and decoder to consolidate the learned features. This helps the model capture abstract representations of the image while balancing computational efficiency.

### 2. **Loss Function**

A custom loss function combining **Binary Cross-Entropy (BCE)** and **Dice Loss** was used:

- **Dice Loss** was selected to directly optimize for overlap between predicted and ground truth regions, crucial for dense segmentation tasks.
- **BCE** complements Dice Loss by focusing on pixel-wise classification, which can stabilize training in cases of class imbalance.
  
This combined loss helped the model learn effectively without over-emphasizing background pixels, which are prevalent in retinal images.

### 3. **Data Augmentation and Oversampling**

To address the limited dataset size, extensive data augmentation was applied using the Albumentations library:

- **Rotation, Flips, and Noise**: These transformations simulate variations in retinal images, allowing the model to generalize better.
- **Random Brightness and Contrast**: Adjustments in brightness and contrast help account for imaging inconsistencies, which can occur due to variations in equipment or lighting.
  
Additionally, an oversampling strategy was employed, with each image being augmented multiple times to increase the dataset size. This approach helped improve model robustness and reduced the risk of overfitting.

### 4. **Normalization as an Option**

Normalization was implemented as an optional step in the transformation pipeline. This flexibility allows for easy comparison of model performance with and without normalization, facilitating fine-tuning based on dataset characteristics. Normalization is better.

### 5. **Regularization**

To prevent overfitting, weight decay was applied during optimization. This regularization technique helped maintain generalization, especially important given the dataset’s limited size.

### 6. **Learning Rate Scheduling**

A learning rate scheduler (`ReduceLROnPlateau`) was used to adjust the learning rate dynamically based on training loss. This scheduling approach prevents the model from becoming stuck in local minima by reducing the learning rate when improvements plateau.

## Results

After training, the model achieved the following test results:

- **Test Dice Coefficient**: 0.8318
- **Test IoU Score**: 0.7121

These scores indicate a high overlap between predicted and ground truth segmentations, with a Dice coefficient over 0.8 and an IoU score above 0.7. This level of accuracy suggests that the model effectively learned the structure of retinal blood vessels and can generalize well on unseen data.

### Result Analysis

- **Dice Coefficient (0.8318)**: This metric reflects a high degree of overlap between the predicted and actual vessel regions, indicating that the model accurately captures vessel boundaries.
- **IoU Score (0.7121)**: The IoU score complements the Dice coefficient by evaluating the intersection-over-union of the segmented vessels. A score above 0.7 suggests reliable segmentation performance, which is particularly valuable for medical applications where precise delineation is necessary.

These results demonstrate the model's ability to provide accurate segmentation on complex medical images, with potential applications in automated diagnostic systems.

## Usage

To train and evaluate the model, follow these steps:

1. **Install Dependencies**: Ensure all required libraries are installed:
   ```bash
   pip install torch albumentations opencv-python-headless