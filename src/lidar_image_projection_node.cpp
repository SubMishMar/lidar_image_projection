#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <tf/transform_listener.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Geometry>

#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

double Rxx, Rxy, Rxz, Ryx, Ryy, Ryz, Rzx, Rzy, Rzz;
double tx, ty, tz;

cv::Mat image_in;
pcl::PCLPointCloud2 *cloud_in = new pcl::PCLPointCloud2;
pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {

    try {
        // converting from ROS to PCL type
        ROS_INFO_STREAM("Converting from ROS to image tyoe");
        image_in = cv_bridge::toCvShare(msg, "bgr8")->image;
        cv::imshow("view", image_in);
        cv::waitKey(30);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    if (msg->width * msg->height == 0) {
        return;
    }
    // converting from ROS to PCL type
    ROS_INFO_STREAM("Converting from ROS to PCL type");
    pcl_conversions::toPCL(*msg, *cloud_in);
    pcl::fromPCLPointCloud2(*cloud_in, *in_cloud);
}

void caminfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    ROS_INFO_STREAM("Listening to Calibration Params");
    double D[5] = {msg->D[0], msg->D[1], msg->D[2], msg->D[3], msg->D[4]};
    double K[9] = {msg->K[0], msg->K[1], msg->K[2], msg->K[3], msg->K[4], msg->K[5], msg->K[6], msg->K[7], msg->K[8]};
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;
    cv::namedWindow("view");
    cv::startWindowThread();
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber image_sub = it.subscribe("/kitti/camera_color_left/image_raw", 1, imageCallback);
    ros::Subscriber cloud_sub = nh.subscribe("/kitti/velo/pointcloud", 1, cloudCallback);
    ros::Subscriber caminfo_sub = nh.subscribe("/kitti/camera_color_left/camera_info", 1, caminfoCallback);

    tf::TransformListener listener;
    tf::StampedTransform transform;
    ros::Rate rate(30);
    while (nh.ok()) {
        try{
            listener.lookupTransform("camera_color_left", "velo_link",
                                     ros::Time(0), transform);

            tx = transform.getOrigin().getX();
            ty = transform.getOrigin().getX();
            tz = transform.getOrigin().getX();

            Eigen::Quaterniond q;
            q.x() = transform.getRotation().getX();
            q.y() = transform.getRotation().getY();
            q.z() = transform.getRotation().getZ();
            q.w() = transform.getRotation().getW();
            Eigen::Matrix3d R = q.normalized().toRotationMatrix();

            Rxx = R(0, 0); Rxy = R(0, 1); Rxz = R(0, 2);
            Ryx = R(1, 0); Ryy = R(1, 1); Ryz = R(1, 2);
            Rzx = R(2, 0); Rzy = R(2, 1); Rzz = R(2, 2);
        }
        catch (tf::TransformException ex){
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
        }
        ros::spinOnce();
        rate.sleep();
    }
    cv::destroyWindow("view");
    return 0;
}
