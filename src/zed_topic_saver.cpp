#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>

namespace fs = std::filesystem;


class ZEDTopicSaver : public rclcpp::Node {
public:
    ZEDTopicSaver()
        : Node("zed_topic_saver") {
        // Declare parameters
        this->declare_parameter<std::string>("output_folder", "zed_output_data");
        this->declare_parameter<std::string>("rgb_topic", "/zed/zed_node/rgb/image_rect_color");
        this->declare_parameter<std::string>("depth_topic", "/zed/zed_node/depth/depth_registered");
        this->declare_parameter<std::string>("pose_topic", "/zed/zed_node/pose");
        this->declare_parameter<std::string>("point_cloud_topic", "/zed/zed_node/mapping/fused_cloud");

        // Get parameters
        output_folder_ = this->get_parameter("output_folder").as_string();
        rgb_topic_ = this->get_parameter("rgb_topic").as_string();
        depth_topic_ = this->get_parameter("depth_topic").as_string();
        pose_topic_ = this->get_parameter("pose_topic").as_string();
        point_cloud_topic_ = this->get_parameter("point_cloud_topic").as_string();

        // Create output directories
        fs::create_directories(output_folder_ + "/rgb");
        fs::create_directories(output_folder_ + "/depth");
        fs::create_directories(output_folder_ + "/point_cloud");
        trajectory_tum_file_.open(output_folder_ + "/trajectory_tum.txt");
        trajectory_replica_file_.open(output_folder_ + "/trajectory_replica.txt");

        // PointCloud Subscriber
        point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            point_cloud_topic_, 10, std::bind(&ZEDTopicSaver::pointCloudCallback, this, std::placeholders::_1));

        // Synchronization setup
        rgb_sub_.subscribe(this, rgb_topic_);
        depth_sub_.subscribe(this, depth_topic_);
        pose_sub_.subscribe(this, pose_topic_);

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), rgb_sub_, depth_sub_, pose_sub_);
        sync_->registerCallback(std::bind(&ZEDTopicSaver::syncCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

        RCLCPP_INFO(this->get_logger(), "ZED Topic Saver Node Initialized with Synchronization");
    }

    ~ZEDTopicSaver() {
        if (trajectory_tum_file_.is_open()) {
            trajectory_tum_file_.close();
        }
        if (trajectory_replica_file_.is_open()) {
            trajectory_replica_file_.close();
        }
    }

private:
    // Synchronized callback for RGB, Depth, and Pose
    void syncCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
        const geometry_msgs::msg::PoseStamped::ConstSharedPtr &pose_msg) {
        try {
            // Save RGB image
            auto rgb_image = cv_bridge::toCvCopy(rgb_msg, "bgr8");
            // set timestamp as nanoseconds
            std::uint64_t timestamp_int = std::uint64_t(rgb_msg->header.stamp.sec * 1e9 + rgb_msg->header.stamp.nanosec);
            std::string timestamp = std::to_string(timestamp_int);
            std::string rgb_filename = output_folder_ + "/rgb/" + timestamp + ".png";
            cv::imwrite(rgb_filename, rgb_image->image);
            RCLCPP_INFO(this->get_logger(), "Saved synchronized RGB image: %s", rgb_filename.c_str());

            // Save Depth image
            auto depth_image = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
            if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
                depth_image->image.convertTo(depth_image->image, CV_16U, 1000);
            }
            std::string depth_filename = output_folder_ + "/depth/" + timestamp + ".png";
            cv::imwrite(depth_filename, depth_image->image);
            RCLCPP_INFO(this->get_logger(), "Saved synchronized Depth image: %s", depth_filename.c_str());

            // Save Pose
            double pose_timestamp = pose_msg->header.stamp.sec + pose_msg->header.stamp.nanosec * 1e-9;
            trajectory_tum_file_ << std::fixed << std::setprecision(9) << pose_timestamp << " "
                                 << pose_msg->pose.position.x << " " << pose_msg->pose.position.y << " " << pose_msg->pose.position.z << " "
                                 << pose_msg->pose.orientation.x << " " << pose_msg->pose.orientation.y << " "
                                 << pose_msg->pose.orientation.z << " " << pose_msg->pose.orientation.w << "\n";

            Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
            transform(0, 3) = pose_msg->pose.position.x;
            transform(1, 3) = pose_msg->pose.position.y;
            transform(2, 3) = pose_msg->pose.position.z;

            Eigen::Quaterniond q(
                pose_msg->pose.orientation.w,
                pose_msg->pose.orientation.x,
                pose_msg->pose.orientation.y,
                pose_msg->pose.orientation.z);
            transform.block<3, 3>(0, 0) = q.toRotationMatrix();

            trajectory_replica_file_ << std::fixed << std::setprecision(9);
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    trajectory_replica_file_ << transform(i, j) << (i == 3 && j == 3 ? "\n" : " ");
                }
            }
            RCLCPP_INFO(this->get_logger(), "Saved synchronized Pose.");
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge Exception: %s", e.what());
        }
    }

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *pcl_cloud);
        
        std::uint64_t timestamp_int = std::uint64_t(msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec);
        std::string timestamp = std::to_string(timestamp_int);
        std::string filename = output_folder_ + "/point_cloud/" + timestamp + ".ply";

        if (pcl::io::savePLYFileBinary(filename, *pcl_cloud) == 0) {
            RCLCPP_INFO(this->get_logger(), "Saved Point Cloud: %s", filename.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to save Point Cloud: %s", filename.c_str());
        }
    }

    // Parameters
    std::string output_folder_;
    std::string rgb_topic_;
    std::string depth_topic_;
    std::string pose_topic_;
    std::string point_cloud_topic_;

    // Message filters
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image, geometry_msgs::msg::PoseStamped>
        SyncPolicy;

    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    message_filters::Subscriber<geometry_msgs::msg::PoseStamped> pose_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // PointCloud Subscriber
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;

    // Output files
    std::ofstream trajectory_tum_file_;
    std::ofstream trajectory_replica_file_;
};


int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ZEDTopicSaver>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
