# ZED2i Topic Saver
Record RGB, Depth, Pose, and Fused Point Cloud from ZED2i

### Dependencies
- ROS2 (tested with humble)
- OpenCV
- PCL

### How to use
1. Install ZED SDK and ROS2 Wrappers from [this official documentation](https://www.stereolabs.com/docs/ros2).
2. Change the config file [common.yaml](https://github.com/stereolabs/zed-ros2-wrapper/blob/master/zed_wrapper/config/common.yaml) to enable mapping and set related parameters.
3. Run zed2 wrapper node for broadcasting topics.
```
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```
4. Run topic saver node
```
ros2 run zed_topic_saver zed_topic_saver --ros-args -p output_folder:=YOURPATH
```
This will save the topics as:
```
YOURPATH/
├── rgb/
│   ├── TIMESTAMP_NS.png
│   └── ...
├── depth/
│   ├── TIMESTAMP_NS.png
│   └── ...
├── point_cloud/
│   ├── TIMESTAMP_NS.ply # XYZRGB
│   └── ...
├── trajectory_replica.txt # each row contains 16 element of 4x4 matrix, 
│                            simlar to replica trajectory
├── trajectory_tum.txt # TUM format trajectory
└── ...
```
Note that ZED yields synced RGB+Depth+Pose topic, and we use message filter (10ms tolerance) for them.
