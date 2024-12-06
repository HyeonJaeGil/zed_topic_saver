import argparse
import numpy as np
import open3d as o3d


def read_tum_trajectory(file_path):
    """
    Reads a TUM trajectory file and extracts timestamps, translation, and rotation (quaternions).
    Returns a list of poses as (translation, rotation) tuples.
    """
    poses = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith("#") or not line.strip():
                    continue
                data = line.strip().split()
                if len(data) != 8:
                    raise ValueError("Invalid line format. Each line must have 8 values: timestamp tx ty tz qx qy qz qw")
                _, tx, ty, tz, qx, qy, qz, qw = map(float, data)
                translation = np.array([tx, ty, tz])
                rotation = np.array([qx, qy, qz, qw])
                poses.append((translation, rotation))
    except Exception as e:
        raise RuntimeError(f"Error reading TUM trajectory file: {e}")
    return poses

def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion to a 3x3 rotation matrix.
    """
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def plot_trajectory_with_open3d(poses):
    """
    Plots the trajectory using Open3D coordinate frames.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for translation, quaternion in poses:
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_to_rotation_matrix(quaternion)

        # Create a coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Apply the pose (rotation + translation)
        coordinate_frame.rotate(rotation_matrix, center=(0, 0, 0))
        coordinate_frame.translate(translation)

        # Add to the visualizer
        vis.add_geometry(coordinate_frame)

    # pcd = o3d.io.read_point_cloud("/home/ori/Dataset/realworld/room2_subsampled/point_cloud/1731954682_180337865.ply")
    # vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot poses from a TUM trajectory file as coordinate frames.")
    parser.add_argument("trajectory_path", type=str, help="Path to the TUM trajectory file.")
    args = parser.parse_args()

    try:
        poses = read_tum_trajectory(args.trajectory_path)
        plot_trajectory_with_open3d(poses)
    except Exception as e:
        print(f"Error: {e}")
