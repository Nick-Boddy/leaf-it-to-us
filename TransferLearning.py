import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import os
from torch.utils.data import Dataset
import cv2
import torch
from torchvision import transforms

import torch.optim as optim
import torch.nn.functional as F

from torchvision import models

import pandas as pd
from plotting import rl_decode

import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

# Constants for image dimensions
HEIGHT = 1400
WIDTH = 875
device = torch.device('mps')



import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class TomatoSegmentationDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None, patch_size=32):
        self.image_dir = image_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
        self.patch_size = patch_size  # Define the patch size

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the image name and construct the image path
        image_name = self.annotations.iloc[idx, 0] + ".jpg"  # Get image ID and append .jpg
        image_path = os.path.join(self.image_dir, image_name)

        # Read the image and convert it to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Get the segmentation mask from the CSV and decode it using run-length decoding
        annotation = self.annotations.iloc[idx, 1]
        mask = rl_decode(annotation)  # Apply the RLE decode function you defined earlier

        # Get the dimensions of the original image and mask
        height, width, _ = image.shape
        assert mask.shape == (height, width), "Image and mask must have the same dimensions"

        # Randomly select the top-left corner of the 32x32 patch
        top_left_x = np.random.randint(0, width - self.patch_size + 1)
        top_left_y = np.random.randint(0, height - self.patch_size + 1)

        # Crop the 32x32 patch from both the image and mask
        image_patch = image[top_left_y:top_left_y + self.patch_size, top_left_x:top_left_x + self.patch_size]
        mask_patch = mask[top_left_y:top_left_y + self.patch_size, top_left_x:top_left_x + self.patch_size]

        if self.transform:
            # Apply the provided transformations to the image patch
            image_patch = self.transform(image_patch)

        # Flatten the mask patch to match the required output format
        mask_patch = torch.tensor(mask_patch, dtype=torch.float32).view(-1)  # Flatten the 32x32 mask to 1D

        # print(image_patch.shape, mask_patch.shape)
        return image_patch, mask_patch


class ResNetSegmentationFFN(nn.Module):
    def __init__(self):
        super(ResNetSegmentationFFN, self).__init__()
        self.resnet = models.resnet34(pretrained=True)  # Load pretrained ResNet
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer

        # Freeze the ResNet layers for the first few epochs
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Add a fully connected layer to output the flattened segmentation map
        self.ffn = nn.Sequential(
            nn.Linear(512, 1024),  # First fully connected layer
            nn.ReLU(),
            nn.Linear(1024, 32 * 32),  # Final fully connected layer that outputs the flattened mask
        )

    def forward(self, x):
        # print(x.shape)
        # Extract features from ResNet (without the fully connected layer)
        x = self.resnet(x)  # Output will be [batch_size, 512, 1, 1]


        # print(x.shape)
        # Flatten the output from ResNet
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 512)


        # print(x.shape)
        # Pass through the FFN to get the flattened segmentation map
        x = self.ffn(x)  # Shape: (batch_size, 1400 * 875)

        return x



# Train function
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_start_time = time.time()

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities


            # print(images.shape,outputs.shape ,masks.shape)
            # Calculate loss
            # print(outputs.shape, masks.shape)
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_duration = time.time() - epoch_start_time
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_duration:.2f}s')

    print('Training complete')

# Evaluate function
def evaluate_model(model, dataloader):
    model.eval()
    dice_score = 0.0
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            print(images.shape, masks.shape)

            # Forward pass
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Convert logits to probabilities

            # Apply threshold to generate binary mask
            preds = (outputs > 0.5).float()

            # Calculate Dice coefficient (or other metrics)
            intersection = (preds * masks).sum()
            dice = (2. * intersection) / (preds.sum() + masks.sum() + 1e-6)
            dice_score += dice.item()

    avg_dice_score = dice_score / len(dataloader)
    print(f'Evaluation Dice Score: {avg_dice_score:.4f}')
    return avg_dice_score

# Call the training and evaluation

def main():

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((HEIGHT, WIDTH)),  # Resize to match image dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # transform = None

    # Assume you have already defined your Dataset and DataLoader
    train_dataset = TomatoSegmentationDataset(image_dir='data/train',
                                              csv_file='data/train.csv',
                                              transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last = True)

    # Define validation dataset and DataLoader for evaluation
    val_dataset = TomatoSegmentationDataset(image_dir='data/test',
                                              csv_file='data/test.csv',
                                            transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last = True)


    # Initialize model, loss function, and optimizer
    model = ResNetSegmentationFFN().to(device)  # num_classes=1 for binary segmentation
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss with logits
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train for 25 epochs
    train_model(model, train_loader, criterion, optimizer, num_epochs=1)

    # Evaluate on validation set
    evaluate_model(model, train_loader)

if __name__ == '__main__':
    main()
