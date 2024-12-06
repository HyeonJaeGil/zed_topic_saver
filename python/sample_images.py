from blurry_image_detector import BlurryImageDetector
from corner_tracker import CornerTracker
import cv2 as cv
import os

if __name__ == '__main__':
    folder = '/home/ori/Dataset/realworld/room_tmp2/rgb'
    image_paths = [os.path.join(folder, image_path)
            for image_path in sorted(os.listdir(folder)) if image_path.endswith('.png')]

    tracker = CornerTracker(image_paths).track()
    error_frames = tracker.get_error_frames()

    blur_detector = BlurryImageDetector(image_paths).detect()
    blurry_frames = blur_detector.get_blurry_frames()

    error_frames = set(error_frames)
    blurry_frames = set(blurry_frames)
    error_frames = error_frames.union(blurry_frames)

    # error_image_paths = [image_paths[i] for i in error_frames]
    # for image_path in error_image_paths:
    #     image = cv.imread(image_path)
    #     cv.imshow('Error', image)
    #     cv.waitKey(0)