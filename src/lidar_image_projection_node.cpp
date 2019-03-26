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
sensor_msgs::PointCloud2 out_cloud;
Eigen::Matrix3d camMat = Eigen::Matrix3d::Identity();
Eigen::MatrixXd C_T_L(3, 4);
Eigen::VectorXd Dist(5);
double fov_x, fov_y;

std::vector<cv::Point2d> lidar_pts_in_fov;

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {

    try {
        // converting from ROS to PCL type
        // ROS_INFO_STREAM("Converting from ROS to image tyoe");
        image_in = cv_bridge::toCvShare(msg, "bgr8")->image;
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }

    if(lidar_pts_in_fov.size() > 0) {
        for(size_t i = 0; i < lidar_pts_in_fov.size(); i++)
            cv::circle(image_in, lidar_pts_in_fov[i], 4, CV_RGB(255, 0, 0), -1, 8, 0);
    } else {
        ROS_WARN("No lidar points in FOV");
    }

    cv::imshow("view", image_in);
    cv::waitKey(30);
}

void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    if (msg->width * msg->height == 0) {
        return;
    }
    // converting from ROS to PCL type
    // ROS_INFO_STREAM("Converting from ROS to PCL type");
    pcl_conversions::toPCL(*msg, *cloud_in);
    pcl::fromPCLPointCloud2(*cloud_in, *in_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_out_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    temp_out_cloud->points.resize(in_cloud->points.size());

    lidar_pts_in_fov.clear();
    for(size_t i = 0; i < in_cloud->points.size(); i++) {

        // Reject points behind the LiDAR
        if(in_cloud->points[i].x < 0)
            continue;
        Eigen::Vector4d pointCloud_L;
        pointCloud_L[0] = in_cloud->points[i].x;
        pointCloud_L[1] = in_cloud->points[i].y;
        pointCloud_L[2] = in_cloud->points[i].z;
        pointCloud_L[3] = 1;

        Eigen::Vector3d pointCloud_C;
        pointCloud_C = C_T_L*pointCloud_L;

        double X = pointCloud_C[0];
        double Y = pointCloud_C[1];
        double Z = pointCloud_C[2];

        double Xangle = atan2(X, Z)*180/CV_PI;
        double Yangle = atan2(Y, Z)*180/CV_PI;

        if(Xangle < -fov_x || Xangle > fov_x)
            continue;

        if(Yangle < -fov_y || Yangle > fov_y)
            continue;

        temp_out_cloud->points[i].x = X;
        temp_out_cloud->points[i].y = Y;
        temp_out_cloud->points[i].z = Z;

        double x_1 = X/Z;
        double y_1 = Y/Z;
        double r = x_1*x_1 + y_1*y_1;

//        x_1 = (x_1 * (1.0 + Dist(0) * r * r + Dist(1) * r * r * r * r +
//                      Dist(4) * r * r * r * r * r * r) +
//               2 * Dist(2) * x_1 * y_1 + Dist(3) * (r * r + 2 * x_1));
//
//        y_1 = (y_1 * (1.0 + Dist(0) * r * r + Dist(1) * r * r * r * r +
//                      Dist(4) * r * r * r * r * r * r) +
//               2 * Dist(3) * x_1 * y_1 + Dist(2) * (r * r + 2 * y_1));

        Eigen::Vector3d x1y1w;
        x1y1w << x_1, y_1, 1;
        Eigen::Vector3d uvw = camMat*x1y1w;
        lidar_pts_in_fov.push_back(cv::Point2d(uvw(0), uvw(1)));
    }
    pcl::PCLPointCloud2 *temp_cloud = new pcl::PCLPointCloud2;
    pcl::toPCLPointCloud2(*temp_out_cloud, *temp_cloud);
    pcl_conversions::fromPCL(*temp_cloud, out_cloud);
    out_cloud.header.stamp = ros::Time::now();
    out_cloud.header.frame_id = "camera_color_left";
}

void caminfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    // ROS_INFO_STREAM("Listening to Calibration Params");
    double D[5] = {msg->D[0], msg->D[1], msg->D[2], msg->D[3], msg->D[4]};
    double K[9] = {msg->K[0], msg->K[1], msg->K[2], msg->K[3], msg->K[4], msg->K[5], msg->K[6], msg->K[7], msg->K[8]};

    camMat << msg->P[0], msg->P[1], msg->P[2],
              msg->P[4], msg->P[5], msg->P[6],
              msg->P[8], msg->P[9], msg->P[10];

    Dist << msg->D[0], msg->D[1], msg->D[2], msg->D[3], msg->D[4];

    fov_x = atan2(msg->height, 2*msg->K[0])*180/CV_PI;
    fov_y = atan2(msg->width, 2*msg->K[4])*180/CV_PI;
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
    ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud_out", 1);
    tf::TransformListener listener;
    tf::StampedTransform transform;
    ros::Rate rate(30);
    while (nh.ok()) {
        try{
            listener.lookupTransform("camera_color_left", "velo_link",
                                     ros::Time(0), transform);

            tx = transform.getOrigin().getX();
            ty = transform.getOrigin().getY();
            tz = transform.getOrigin().getZ();

            Eigen::Quaterniond q;
            q.x() = transform.getRotation().getX();
            q.y() = transform.getRotation().getY();
            q.z() = transform.getRotation().getZ();
            q.w() = transform.getRotation().getW();
            Eigen::Matrix3d R = q.normalized().toRotationMatrix();

            Rxx = R(0, 0); Rxy = R(0, 1); Rxz = R(0, 2);
            Ryx = R(1, 0); Ryy = R(1, 1); Ryz = R(1, 2);
            Rzx = R(2, 0); Rzy = R(2, 1); Rzz = R(2, 2);

            C_T_L << Rxx, Rxy, Rxz, tx,
                     Ryx, Ryy, Ryz, ty,
                     Rzx, Rzy, Rzz, tz;
        }
        catch (tf::TransformException ex){
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
        }
        cloud_pub.publish(out_cloud);
        ros::spinOnce();
        rate.sleep();
    }
    cv::destroyWindow("view");
    return 0;
}
