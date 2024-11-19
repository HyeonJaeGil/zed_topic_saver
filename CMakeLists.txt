cmake_minimum_required(VERSION 3.8)
project(zed_topic_saver)

# Find dependencies
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(pcl_ros REQUIRED)

# Create executable
add_executable(zed_topic_saver src/zed_topic_saver.cpp)
target_link_libraries(zed_topic_saver
    ${PCL_LIBRARIES}
)

# Link libraries
ament_target_dependencies(zed_topic_saver
    rclcpp
    sensor_msgs
    geometry_msgs
    cv_bridge
    OpenCV
    pcl_conversions
)

install(TARGETS zed_topic_saver DESTINATION lib/${PROJECT_NAME})
ament_package()
