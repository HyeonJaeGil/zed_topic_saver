import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def plot_laplacian_variance(folder_path, neighbor_window=5, threshold_ratio=0.7):
    # Get all image filenames
    images = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    laplacian_variances = []
    discarded_indices = []

    # Compute Laplacian variance for all frames
    for image_path in tqdm(images, ncols=80):
        image = cv2.imread(image_path)
        if image is not None:
            laplacian_variances.append(compute_laplacian_variance(image))

    # Check consistency and mark discarded frames
    for i in range(len(laplacian_variances)):
        start = max(0, i - neighbor_window)
        end = min(len(laplacian_variances), i + neighbor_window + 1)
        local_mean = np.mean(laplacian_variances[start:end])

        if laplacian_variances[i] < threshold_ratio * local_mean:
            discarded_indices.append(i)

    # Plot the Laplacian variances
    plt.figure(figsize=(12, 6))
    plt.plot(laplacian_variances, label='Laplacian Variance', color='blue', marker='o', markersize=4)
    plt.scatter(discarded_indices, [laplacian_variances[i] for i in discarded_indices],
                color='red', label='Discarded Frames', zorder=5)

    plt.title('Laplacian Variance Across Frames')
    plt.xlabel('Frame Index')
    plt.ylabel('Laplacian Variance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    discarded_images = [images[index] for index in discarded_indices]
    return laplacian_variances, discarded_images


def visualize_discarded_frames(discarded_frames, max_frames=20):
    plt.figure(figsize=(15, 8))
    for i, frame_path in enumerate(discarded_frames[:max_frames]):
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(4, 5, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Folder path and parameters
folder_path = "/home/ori/Dataset/realworld/room2/rgb"
neighbor_window = 5
threshold_ratio = 0.7

# Plot Laplacian Variance
laplacian_variances, discarded_images = plot_laplacian_variance(folder_path, neighbor_window, threshold_ratio)

visualize_discarded_frames(discarded_images)

