import os
import shutil
from blurry_image_detector import BlurryImageDetector
from corner_tracker import CornerTracker


if __name__ == '__main__':

    # input folder to subsample
    input_folder = '/home/ori/Dataset/realworld/room2'
    rgb_folder = os.path.join(input_folder, 'rgb')
    depth_folder = os.path.join(input_folder, 'depth')
    pose_file = os.path.join(input_folder, 'trajectory_kitti.txt')
    with open(pose_file, 'r') as f:
        poses = f.readlines()
    
    rgb_paths = [os.path.join(rgb_folder, image_path)
            for image_path in sorted(os.listdir(rgb_folder)) if image_path.endswith('.png')]
    
    ### HACK: remove files whose filename is bigger than 1731954677165738925
    discard_frames = [i for i in range(len(rgb_paths)) if int(os.path.basename(rgb_paths[i]).split('.')[0]) >= 1731954677165738925]
    rgb_paths = [rgb_paths[i] for i in range(len(rgb_paths)) if i not in discard_frames]

    tracker = CornerTracker(rgb_paths).track()
    error_frames = tracker.get_error_frames()

    blur_detector = BlurryImageDetector(rgb_paths, threshold=80).detect()
    blurry_frames = blur_detector.get_blurry_frames()

    stride = 1
    skip_frames = [i for i in range(len(rgb_paths)) if i % stride != 0]

    final_discard_frames = set(discard_frames + error_frames + blurry_frames + skip_frames)
    print(f'Discarded {len(final_discard_frames)} frames')

    # specify output folder and the strides to subsample
    output_folder = '/home/ori/Dataset/realworld/room2_subsampled_small'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'depth'), exist_ok=True)

    rgb_subsampled = [os.path.basename(path) for i, path in enumerate(rgb_paths) if i not in final_discard_frames]
    depth_subsampled = [os.path.join(depth_folder, os.path.basename(path)) for path in rgb_subsampled]
    poses_subsampled = [pose for i, pose in enumerate(poses) if i not in final_discard_frames]

    for i in range(len(rgb_subsampled)):
        shutil.copy2(os.path.join(rgb_folder, rgb_subsampled[i]), os.path.join(output_folder, 'rgb', f'{i:06d}.png'))
        shutil.copy2(os.path.join(depth_folder, depth_subsampled[i]), os.path.join(output_folder, 'depth', f'{i:06d}.png'))
    with open(os.path.join(output_folder, 'trajectory_kitti.txt'), 'w') as f:
        f.writelines(poses_subsampled)

    print(f'Subsampled {len(rgb_subsampled)} frames to {output_folder}')
