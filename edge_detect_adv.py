import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from rl_decode import *
# from score import *



def load_images_and_labels(dataset_folder='data', train=True):
    """
    Load all images and their corresponding segmentation labels from the dataset,
    and save them as NumPy arrays.

    Args:
        dataset_folder (str): Path to the dataset folder.
        train (bool): Whether to load training or test data.

    Returns:
        tuple: (images, labels, image_ids) where:
            - images is a NumPy array of all images,
            - labels is a NumPy array of corresponding segmentation masks (None for test data),
            - image_ids is a list of image filenames/IDs.
    """
    if train:
        image_folder = os.path.join(dataset_folder, 'train')
        csv_path = os.path.join(dataset_folder, 'train.csv')
    else:
        image_folder = os.path.join(dataset_folder, 'test')
        csv_path = os.path.join(dataset_folder, 'test.csv')

    # Load segmentation labels from CSV
    segmentation_data = pd.read_csv(csv_path)

    # Create lists to hold images, labels, and image IDs
    images = []
    labels = []
    image_ids = []

    # Loop over all images in the folder
    for image_name in segmentation_data['id']:
        image_path = os.path.join(image_folder, image_name + '.jpg')

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Image {image_name} not found.")
            continue

        # Extract the segmentation label and decode it (if in training mode)
        if train:
            seg_info = segmentation_data[segmentation_data['id'] == image_name]
            encoded_label = seg_info['annotation'].values[0]
            segmentation_mask = rl_decode(encoded_label)
        else:
            segmentation_mask = None

        # Append the image, label, and image ID to the lists
        images.append(image)
        labels.append(segmentation_mask)
        image_ids.append(image_name)

    # Convert lists to NumPy arrays
    images_np = np.array(images)
    labels_np = np.array(labels)

    # Return the images, labels, and image IDs
    return images_np, labels_np, image_ids


def visualize_and_save_images(images_np, labels_np=None, labels_alpha = 0.5, save_path='train_images_plot'):
    """
    Visualize all 17 training images in a single plot with 9 images per row,
    and optionally mark the regions where labels == 1 on the figure.

    Args:
        images_np (numpy.ndarray): NumPy array of images of shape (N, H, W, C).
        labels_np (numpy.ndarray, optional): NumPy array of labels of shape (N, H, W).
        save_path (str): Path where the generated plot will be saved.
    """
    # # Check that we have at least 17 images
    # if images_np.shape[0] < 17:
    #     raise ValueError("There must be at least 17 images in the dataset.")
    #
    # # Check if labels are provided and ensure labels match the number of images
    # if labels_np is not None and labels_np.shape[0] < 17:
    #     raise ValueError("There must be at least 17 labels if labels are provided.")

    # Set up the plot (we want 9 images per row and ceil(17/9) rows, i.e., 2 rows)
    fig, axs = plt.subplots(2, 9, figsize=(18, 6), dpi = 500)  # 2 rows, 9 columns

    # Loop over the first 17 images
    for i in range(len(images_np)):
        row = i // 9  # Determine the row number
        col = i % 9   # Determine the column number
        ax = axs[row, col]  # Get the axis for the specific subplot

        # Display the image
        ax.imshow(cv2.cvtColor(images_np[i], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display

        # If labels are provided, overlay the segmentation mask
        if labels_np is not None:
            mask = labels_np[i]

            white_mask = np.zeros_like(images_np[i], dtype=np.uint8)
            if white_mask.ndim == 3:  # For RGB images
                white_mask[mask == 1] = [255, 255, 255]
            elif white_mask.ndim == 2:  # For grayscale images
                white_mask[mask == 1] = 255 # Set the mask area to white

            # Overlay the white mask with transparency
            ax.imshow(white_mask,  cmap='gray' if white_mask.ndim == 2 else None, alpha=labels_alpha)

        ax.axis('off')  # Hide the axis

    # Remove any extra subplots (since we only have 17 images)
    for j in range(17, 18):
        fig.delaxes(axs[j // 9, j % 9])

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(save_path + '.pdf', format = 'pdf', bbox_inches = 'tight')
    plt.show()

def jaccard_score(pred, target, smooth=1e-6):
    """
    Compute the Jaccard loss, which is 1 - Jaccard score.

    Args:
        pred (torch.Tensor): Predicted edges (binary mask) of shape (N, H, W).
        target (torch.Tensor): Ground truth segmentation mask (binary mask) of shape (N, H, W).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: The Jaccard loss between the predicted and ground truth masks.
    """
    # Flatten the tensors to 1D arrays (N * H * W)
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Compute the intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    # Compute the Jaccard score
    jaccard_score = (intersection + smooth) / (union + smooth)

    # Return the Jaccard loss
    return jaccard_score


def preprocess_images(images_np, black_threshold=50):
    """
    Preprocess each image in images_np by detecting black backgrounds,
    recoloring them with the average foreground color, and applying Gaussian blur.

    Args:
        images_np (np.ndarray): NumPy array of images of shape (N, H, W, C).
        black_threshold (int): Threshold to detect black background pixels.

    Returns:
        np.ndarray: Preprocessed images, in grayscale, ready for edge detection.
    """
    # List to store the preprocessed images
    # preprocessed_images = [cv2.GaussianBlur(i, (5, 5), 0) for i in images_np]
    preprocessed_images = [cv2.bilateralFilter(img,9,75,75) for img in images_np]

    # Convert the list of preprocessed images back to a NumPy array
    return np.array(preprocessed_images)

def edge_detect(images_np):
    """
    Process each image by running edge detection and save the processed edges
    and labels into NumPy arrays.

    Args:
        images_np (numpy.ndarray): NumPy array of images of shape (N, H, W, C).
        labels_np (numpy.ndarray, optional): NumPy array of labels of shape (N, H, W).

    Returns:
        tuple: (edges_np, labels_np) where edges_np contains the edge-detected images,
               and labels_np contains the corresponding segmentation masks (if provided).
    """
    # Create an empty list to store the edge-detected images
    edges_list = [cv2.Canny(i, threshold1=100, threshold2=200) for i in images_np]

    # Convert the list of edges into a NumPy array
    edges_np = np.array(edges_list)

    edges_np[edges_np > 0] = 1

    return edges_np


def detect_black_background(image, black_threshold=10):
    """
    Detect the black background in an image. A pixel is considered part of the black background
    if all color channels have values below the black_threshold.

    Args:
        image (np.ndarray): Input image in BGR or RGB format.
        black_threshold (int): Threshold to detect black pixels. Default is 10.

    Returns:
        np.ndarray: A binary mask where 1 indicates the background (black) and 0 indicates the foreground.
    """
    # Detect black pixels by checking if all channels are below the black_threshold
    mask_black = np.all(image < black_threshold, axis=-1)

    # Convert the mask to uint8 (0 and 1)
    return mask_black.astype(np.uint8)

def smooth_background(background_mask, kernel_size=5):
    """
    Smooth the detected background by expanding it within the range of the kernel size.
    This ensures that if any part of a kernel contains background, the entire kernel
    area is labeled as background.

    Args:
        background_mask (np.ndarray): Binary mask where 1 indicates background, 0 indicates foreground.
        kernel_size (int): Size of the structuring element (kernel) used to smooth the background.

    Returns:
        np.ndarray: Smoothed background mask.
    """
    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply dilation to expand the background areas
    smoothed_background = cv2.dilate(background_mask, kernel, iterations=1)

    return smoothed_background

def filter_edges_in_background(edges, background):
    """
    Filter out edges that fall within the background of the image.

    Args:
        edges (np.ndarray): Binary edge-detected image (0 and 1 values or 0 and 255).

    Returns:
        np.ndarray: Filtered edge image with edges removed in the background areas.
    """
    # Ensure edges are in the same format as background_mask (binary)
    edges_binary = (edges > 0).astype(np.uint8)

    # Mask out the edges that fall within the background
    filtered_edges = np.where(background == 1, 0, edges_binary)

    return filtered_edges


def close_edges(edges, kernel_size=3):
    """
    Apply morphological closing to boost connectivity between edges by closing gaps.

    Args:
        edges (np.ndarray): Binary edge-detected image (0 and 1 values or 0 and 255).
        kernel_size (int): Size of the structuring element (kernel) for closing.

    Returns:
        np.ndarray: Closed edges to improve connectivity.
    """
    # Create a kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply closing (dilation followed by erosion)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return closed_edges



def edge_detection(images, labels = [], visualization = False):
    # Assuming `image_np` is a NumPy array of an image
    black_background_mask = detect_black_background(images, black_threshold=25)
    print('\nBackground Detection Done!')
    print("Image shape:", images.shape)
    print("black_background_mask shape:", black_background_mask.shape)
    if visualization:
        visualize_and_save_images(images, labels_np=black_background_mask, save_path='background_mask', labels_alpha = 1)

    # Assuming `image_np` is a NumPy array of an image
    black_background_mask = np.array([smooth_background(i, kernel_size=20) for i in black_background_mask])
    print('\nBackground Dilation Done!')
    print("Image shape:", images.shape)
    print("black_background_mask shape:", black_background_mask.shape)
    if visualization:
        visualize_and_save_images(images, labels_np=black_background_mask, save_path='background_mask_dilation', labels_alpha = 1)

    edges = edge_detect(images)
    print('\nCanny Edge Detection Done!')
    print("Image shape:", images.shape)
    print("edges shape:", edges.shape)
    if visualization:
        visualize_and_save_images(images, labels_np=edges, save_path='canny_prediction', labels_alpha = 1)
    if len(labels) > 0:
        loss = jaccard_score(edges, labels)
        print(f'Jaccard Loss: {loss.item()}')

    edges = filter_edges_in_background(edges, black_background_mask)
    print('\nBackground Edge removal Done!')
    print("Image shape:", images.shape)
    print("edges shape:", edges.shape)
    if visualization:
        visualize_and_save_images(images, labels_np=edges, save_path='canny_prediction_nobackground', labels_alpha = 1)
    if len(labels) > 0:
        loss = jaccard_score(edges, labels)
        print(f'Jaccard Loss: {loss.item()}')

    edges = np.array([close_edges(i, kernel_size = 5) for i in edges])
    print('\nEdge closing Done!')
    print("Image shape:", images.shape)
    print("edges shape:", edges.shape)
    if visualization:
        visualize_and_save_images(images, labels_np=edges, save_path='canny_prediction_close', labels_alpha = 1)
    if len(labels) > 0:
        loss = jaccard_score(edges, labels)
        print(f'Jaccard Loss: {loss.item()}')

    return edges


def write_edges_to_csv(edges, images_id, output_csv='prediction.csv'):

    # List to store the results
    results = []

    # Iterate over the edges and corresponding images
    for i, (edge_map, image_id) in enumerate(zip(edges, images_id)):
        # Run-length encode the edge map
        rle_encoded_edges = rl_encode(edge_map)

        # Append the image id and encoded edges to the results
        results.append({'id': image_id, 'annotation': rle_encoded_edges})

    # Convert the results to a DataFrame and save as a CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print(f'Encoded edges saved to {output_csv}')

print('Train Dataset')
images, labels, image_ids = load_images_and_labels(train=True)
edges = edge_detection(images, labels = labels, visualization = False)

print('Test Dataset')
images, labels, image_ids = load_images_and_labels(train=False)
edges = edge_detection(images, visualization = False)
write_edges_to_csv(edges, image_ids, output_csv='prediction.csv')
