import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants for image dimensions
HEIGHT = 1400
WIDTH = 875

# Function for run-length decoding
def rl_decode(enc):
    parts = [int(s) for s in enc.split(' ')]
    dec = list()
    for i in range(0, len(parts), 2):
        cnt = parts[i]
        val = parts[i + 1]
        dec += cnt * [val]
    return np.array(dec, dtype=np.uint8).reshape((HEIGHT, WIDTH))

# Function to plot the image and its segmentation mask
def plot_image_and_segmentation(image_name, dataset_folder):
    # Set paths for images and CSV
    image_path = os.path.join(dataset_folder, 'train', image_name + '.jpg')
    csv_path = os.path.join(dataset_folder, 'train.csv')

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image {image_name} not found in the dataset folder.")

    # Convert the image from BGR to RGB (OpenCV loads images in BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the CSV containing segmentation labels
    segmentation_data = pd.read_csv(csv_path)

    # Assuming the CSV contains two columns: 'image_name' and 'segmentation_label'
    seg_info = segmentation_data[segmentation_data['id'] == image_name]

    if seg_info.empty:
        raise ValueError(f"No segmentation data found for {image_name} in the CSV.")

    # Extract the segmentation label and decode it using rl_decode
    encoded_label = seg_info['annotation'].values[0]
    segmentation_mask = rl_decode(encoded_label)

    # Plot the image and its segmentation mask
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image
    ax[0].imshow(image_rgb)
    ax[0].set_title(f'Original Image: {image_name}')
    ax[0].axis('off')

    # Plot the segmentation mask
    ax[1].imshow(segmentation_mask, cmap='gray')
    ax[1].set_title(f'Segmentation Mask: {image_name}')
    ax[1].axis('off')

    plt.show()

# Example usage
# dataset_folder = 'data'
# image_name = 'leaf01'  # Replace with actual image name
# plot_image_and_segmentation(image_name, dataset_folder)
