import argparse
import numpy as np
import cv2

def visualize_depth_image(image_path):
    # Read the depth image
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Min-max normalization
    depth_min = np.min(depth_image)
    depth_max = np.max(depth_image)
    normalized_depth = (depth_image - depth_min) / (depth_max - depth_min)
    normalized_depth = np.clip(normalized_depth, 0, 1)  # Clip values to [0, 1]

    # Scale to 0-255 for visualization
    display_image = (normalized_depth * 255).astype(np.uint8)

    # Show the image using OpenCV
    cv2.imshow('Normalized Depth Image', display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a depth image with min-max normalization.")
    parser.add_argument("image_path", type=str, help="Path to the depth image file.")
    args = parser.parse_args()

    try:
        visualize_depth_image(args.image_path)
    except Exception as e:
        print(f"Error: {e}")
