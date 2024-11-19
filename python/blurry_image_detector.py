import cv2
import os
import numpy as np
from tqdm import tqdm

class BlurryImageDetector:
    def __init__(self, image_paths, threshold=100.0):
        """
        Initialize the BlurryImageDetector class.

        Args:
            image_paths (list[str]): List of image paths.
            threshold (float): Variance threshold to classify images as blurry.
        """
        self.image_paths = image_paths
        self.threshold = threshold
        self.laplacian_variances = []
        self.blurry_frames = []

    def _compute_laplacian_variance(self, image):
        """
        Compute the Laplacian variance of an image as a sharpness measure.

        Args:
            image (np.ndarray): Input image.

        Returns:
            float: Variance of the Laplacian.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def detect(self):
        """
        Detect blurry frames in the folder.

        Returns:
            self: Returns the instance for method chaining.
        """
        self.laplacian_variances.clear()
        self.blurry_frames.clear()

        for idx, image_path in enumerate(tqdm(self.image_paths, ncols=80)):
            image = cv2.imread(image_path)
            if image is None:
                continue
            laplacian_var = self._compute_laplacian_variance(image)
            self.laplacian_variances.append((idx, laplacian_var))

            if laplacian_var < self.threshold:
                self.blurry_frames.append(idx)

        return self

    def get_blurry_frames(self):
        """
        Get the list of indices for blurry frames.

        Returns:
            list[int]: Indices of blurry frames.
        """
        return self.blurry_frames

    def get_laplacian_variances(self):
        """
        Get the Laplacian variances for all frames.

        Returns:
            list[tuple]: List of tuples containing frame indices and their Laplacian variances.
        """
        return self.laplacian_variances

if __name__ == '__main__':

    folder = 'YOUR_FOLDER_PATH'
    image_paths = [os.path.join(folder, image_path)
            for image_path in sorted(os.listdir(folder)) if image_path.endswith('.png')]

    blur_detector = BlurryImageDetector(image_paths).detect()

    blurry_frames = blur_detector.get_blurry_frames()
    print(f"Blurry frame indices: {blurry_frames}")
    blurry_image_paths = [image_paths[i] for i in blurry_frames]
    for image_path in blurry_image_paths:
        image = cv2.imread(image_path)
        cv2.imshow('Blurred', image)
        cv2.waitKey(0)