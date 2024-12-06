import open3d as o3d
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Point Cloud")
    parser.add_argument("path", type=str, help="Path to the point cloud file")
    args = parser.parse_args()

    # check if path is a folder or a file
    if os.path.isdir(args.path):
        paths = [os.path.join(args.path, file) for file in sorted(os.listdir(args.path))
                 if file.endswith('.ply') or file.endswith('.pcd')]
    else:
        paths = [args.path]
    
    for path in paths:
        print(f"Visualizing {path}")
        pcd = o3d.io.read_point_cloud(path)
        o3d.visualization.draw_geometries([pcd],
                                        zoom=0.8,
                                        front=[-0.42499054040951256, 0.64103502964216752, 0.63910651016406961],
                                        lookat=[1.3461292256564157, -0.59203721781134633, -1.1847040907832462],
                                        up=[0.35188833893870192, -0.53351015284360837, 0.76911736018097465]
        )

