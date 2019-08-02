#include <algorithm>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <tf/transform_listener.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Geometry>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

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

#include <pcl/filters/passthrough.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>

#include <pcl/filters/statistical_outlier_removal.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <iostream>
#include <fstream>

#include <random>

#include <Image.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
        sensor_msgs::Image> SyncPolicy;

class lidarImageProjection {
private:

    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud_sub;
    message_filters::Subscriber<sensor_msgs::Image> *image_sub;
    message_filters::Synchronizer<SyncPolicy> *sync;

    ros::Publisher cloud_pub;
    ros::Publisher image_pub;

    cv::Mat c_R_l, tvec;
    cv::Mat rvec;
    std::string result_str;
    Eigen::Matrix4d C_T_L, L_T_C;
    Eigen::Matrix3d C_R_L, L_R_C;
    Eigen::Quaterniond C_R_L_quatn, L_R_C_quatn;
    Eigen::Vector3d C_t_L, L_t_C;

    bool project_only_plane;
    cv::Mat projection_matrix;
    cv::Mat distCoeff;
    double bf;

    bool color_projection;

    std::vector<cv::Point3d> objectPoints_L, objectPoints_C;
    std::vector<cv::Point2d> imagePoints;

    sensor_msgs::PointCloud2 out_cloud_ros;

    std::string lidar_frameId;

    std::string camera_in_topic;
    std::string lidar_in_topic;

    pcl::PointCloud<pcl::PointXYZRGB> out_cloud_pcl;
    cv::Mat image_in, image_out;

    int dist_cut_off;
    int frame_count;

    std::string cam_config_file_path;
    int image_width, image_height;

    std::string camera_name;

    double stddev_rot, stddev_trans;
    cv::Mat image_projected;
    int frame_no;

    double corr_score;

    std::vector<Frame> dataFrames;

public:
    lidarImageProjection() {
        camera_in_topic = readParam<std::string>(nh, "camera_in_topic");
        lidar_in_topic = readParam<std::string>(nh, "lidar_in_topic");
        dist_cut_off = readParam<int>(nh, "dist_cut_off");
        camera_name = readParam<std::string>(nh, "camera_name");
        cloud_sub =  new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, lidar_in_topic, 1);
        image_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh, camera_in_topic, 1);
        std::string lidarOutTopic = camera_in_topic + "/velodyne_out_cloud";
        cloud_pub = nh.advertise<sensor_msgs::PointCloud2>(lidarOutTopic, 1);
        std::string imageOutTopic = camera_in_topic + "/projected_image";
        image_pub = nh.advertise<sensor_msgs::Image>(imageOutTopic, 1);

        sync = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *cloud_sub, *image_sub);
        sync->registerCallback(boost::bind(&lidarImageProjection::callback, this, _1, _2));

        C_T_L = Eigen::Matrix4d::Identity();
        c_R_l = cv::Mat::zeros(3, 3, CV_64F);
        tvec = cv::Mat::zeros(3, 1, CV_64F);

        result_str = readParam<std::string>(nh, "result_file");
        project_only_plane = readParam<bool>(nh, "project_only_plane");
        color_projection = readParam<bool>(nh, "color_projection");
        projection_matrix = cv::Mat::zeros(3, 3, CV_64F);
        distCoeff = cv::Mat::zeros(5, 1, CV_64F);

        stddev_rot = readParam<double>(nh, "stddev_rot");
        stddev_trans = readParam<double>(nh, "stddev_trans");

        std::ifstream myReadFile(result_str.c_str());
        std::string word;
        int i = 0;
        int j = 0;
        while (myReadFile >> word){
            C_T_L(i, j) = atof(word.c_str());
            j++;
            if(j>3) {
                j = 0;
                i++;
            }
        }

        addGaussianNoise(C_T_L);

        L_T_C = C_T_L.inverse();

        C_R_L = C_T_L.block(0, 0, 3, 3);
        C_t_L = C_T_L.block(0, 3, 3, 1);

        L_R_C = L_T_C.block(0, 0, 3, 3);
        L_t_C = L_T_C.block(0, 3, 3, 1);

        cv::eigen2cv(C_R_L, c_R_l);
        C_R_L_quatn = Eigen::Quaterniond(C_R_L);
        L_R_C_quatn = Eigen::Quaterniond(L_R_C);
        cv::Rodrigues(c_R_l, rvec);
        cv::eigen2cv(C_t_L, tvec);

        cam_config_file_path = readParam<std::string>(nh, "cam_config_file_path");
        readCameraParams(cam_config_file_path,
                         image_height,
                         image_width,
                         distCoeff,
                         projection_matrix);
        frame_no = 0;
        frame_count = 0;
        corr_score = 0.0;
    }

    void addGaussianNoise(Eigen::Matrix4d &transformation) {
        std::vector<double> data_rot = {0, 0, 0};
        const double mean_rot = 0.0;
        std::default_random_engine generator_rot;
        std::normal_distribution<double> dist(mean_rot, stddev_rot);

        // Add Gaussian noise
        for (auto& x : data_rot) {
            x = x + dist(generator_rot);
        }

        // Output the result, for demonstration purposes
        double roll = data_rot[0]*M_PI/180;
        double pitch = data_rot[1]*M_PI/180;
        double yaw = data_rot[2]*M_PI/180;
//        ROS_WARN_STREAM("Roll: " << roll << " Pitch: " << pitch << " Yaw: " << yaw);

        Eigen::Matrix3d m;
        m = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())
            * Eigen::AngleAxisd(pitch,  Eigen::Vector3d::UnitY())
            * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

        std::vector<double> data_trans = {0, 0, 0};
        const double mean_trans = 0.0;
        std::default_random_engine generator_trans;
        std::normal_distribution<double> dist_trans(mean_trans, stddev_trans);

        // Add Gaussian noise
        for (auto& x : data_trans) {
            x = x + dist_trans(generator_trans);
        }

        // Output the result, for demonstration purposes
        Eigen::Vector3d trans;
        trans(0) = data_trans[0];
        trans(1) = data_trans[1];
        trans(2) = data_trans[2];
//        ROS_WARN_STREAM("X: " << trans.x() << " Y: " << trans.y() << " Z: " << trans.z());

        Eigen::Matrix4d trans_noise = Eigen::Matrix4d::Identity();
        trans_noise.block(0, 0, 3, 3) = m;
        trans_noise.block(0, 3, 3, 1) = trans;
        transformation = transformation*trans_noise;
    }

    void readCameraParams(std::string cam_config_file_path,
                          int &image_height,
                          int &image_width,
                          cv::Mat &D,
                          cv::Mat &K) {
        cv::FileStorage fs_cam_config(cam_config_file_path, cv::FileStorage::READ);
        if(!fs_cam_config.isOpened())
            std::cerr << "Error: Wrong path: " << cam_config_file_path << std::endl;
        fs_cam_config["image_height"] >> image_height;
        fs_cam_config["image_width"] >> image_width;
        fs_cam_config["k1"] >> D.at<double>(0);
        fs_cam_config["k2"] >> D.at<double>(1);
        fs_cam_config["p1"] >> D.at<double>(2);
        fs_cam_config["p2"] >> D.at<double>(3);
        fs_cam_config["k3"] >> D.at<double>(4);
        fs_cam_config["fx"] >> K.at<double>(0, 0);
        fs_cam_config["fy"] >> K.at<double>(1, 1);
        fs_cam_config["cx"] >> K.at<double>(0, 2);
        fs_cam_config["cy"] >> K.at<double>(1, 2);
        fs_cam_config["bf"] >> bf;
    }

    template <typename T>
    T readParam(ros::NodeHandle &n, std::string name){
        T ans;
        if (n.getParam(name, ans)){
            ROS_INFO_STREAM("Loaded " << name << ": " << ans);
        } else {
            ROS_ERROR_STREAM("Failed to load " << name);
            n.shutdown();
        }
        return ans;
    }

    cv::Vec3b atf(cv::Mat rgb, cv::Point2d xy_f){
        cv::Vec3i color_i;
        color_i.val[0] = color_i.val[1] = color_i.val[2] = 0;

        int x = xy_f.x;
        int y = xy_f.y;

        for (int row = 0; row <= 1; row++){
            for (int col = 0; col <= 1; col++){
                if((x+col)< rgb.cols && (y+row) < rgb.rows) {
                    cv::Vec3b c = rgb.at<cv::Vec3b>(cv::Point(x + col, y + row));
                    for (int i = 0; i < 3; i++){
                        color_i.val[i] += c.val[i];
                    }
                }
            }
        }

        cv::Vec3b color;
        for (int i = 0; i < 3; i++){
            color.val[i] = color_i.val[i] / 4;
        }
        return color;
    }

    void publishTransforms() {
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        tf::quaternionEigenToTF(L_R_C_quatn, q);
        transform.setOrigin(tf::Vector3(L_t_C(0), L_t_C(1), L_t_C(2)));
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), lidar_frameId, camera_name));
    }

    void colorPointCloud() {
        out_cloud_pcl.points.clear();
        out_cloud_pcl.resize(objectPoints_L.size());

        for(size_t i = 0; i < objectPoints_L.size(); i++) {
            cv::Vec3b rgb = atf(image_in, imagePoints[i]);
            pcl::PointXYZRGB pt_rgb(rgb.val[2], rgb.val[1], rgb.val[0]);
            pt_rgb.x = objectPoints_L[i].x;
            pt_rgb.y = objectPoints_L[i].y;
            pt_rgb.z = objectPoints_L[i].z;
            out_cloud_pcl.push_back(pt_rgb);
        }
    }

    void findSimilarity(std::vector<cv::Point2i> edge_points, cv::Mat idt_edge_img) {
        double score = 0;
        for(int i = 0; i < edge_points.size(); i++) {
            int u = edge_points[i].x;
            int v = edge_points[i].y;
            score += (double)idt_edge_img.at<uchar>(v, u);
        }
        ROS_WARN_STREAM("Similarity Score: " << score/(double)edge_points.size());
    }

    void correlation(cv::Mat image_1, cv::Mat image_2) {
//        // convert data-type to "float"
//        cv::Mat im_float_1;
//        image_1.convertTo(im_float_1, CV_32F);
//        cv::Mat im_float_2;
//        image_2.convertTo(im_float_2, CV_32F);
//        ROS_ASSERT(image_1.type() == image_2.type());
        cv::Mat output;
        cv::matchTemplate(image_1, image_2, output, CV_TM_CCOEFF_NORMED);
        double abs_corr_score = fabs(output.at<float>(0));
        corr_score += abs_corr_score;
        ROS_INFO_STREAM("Corr Score: " << abs_corr_score);
    }

    void colorLidarPointsOnImage(double min_range,
                                 double max_range,
                                 double min_height,
                                 double max_height) {
        double error = 0;
        double count = 0;
        Image::Image img(image_in);
        cv::Mat image_edge_gray = img.computeIDTEdgeImage();
        cv::imshow("idt edge image", image_edge_gray);
        cv::waitKey(1);
        cv::Mat image_edge_color;
        cv::cvtColor(image_edge_gray, image_edge_color, CV_GRAY2BGR);
        std::vector<cv::Point2i> edge_points_lidar;
        cv::Mat image_lidar_pts = cv::Mat::zeros(image_in.size(), image_out.type());
        for(size_t i = 0; i < imagePoints.size(); i++) {
            int u = imagePoints[i].x;
            int v = imagePoints[i].y;

            double X = objectPoints_C[i].x;
            double Y = objectPoints_C[i].y;
            double Z = objectPoints_C[i].z;
//            double range = sqrt(X*X + Y*Y + Z*Z);
            double range = Z;
            int d = image_in.at<uchar>(v, u);
            if(d <= 0)
                continue;
            edge_points_lidar.push_back(cv::Point2i(u, v));
            image_out = image_edge_color;
            if(color_projection) {
                double red_field = 255*(range - min_range)/(max_range - min_range);
                double green_field = 255*(max_range - range)/(max_range - min_range);
                double blue_field = 255*(Z - min_height)/(max_height - min_height);
                cv::circle(image_out, cv::Point2i(u, v), 1,
                           CV_RGB(red_field, green_field, blue_field), -1, 1, 0);
                cv::circle(image_lidar_pts, cv::Point2i(u, v), 1,
                           CV_RGB(red_field, green_field, blue_field), -1, 1, 0);
            } else {
                double red_field = 255;
                double green_field = 255;
                double blue_field = 255;
                cv::circle(image_out, cv::Point2i(u, v), 1,
                           CV_RGB(red_field, green_field, blue_field), -1, 1, 0);
                cv::circle(image_lidar_pts, cv::Point2i(u, v), 1,
                           CV_RGB(red_field, green_field, blue_field), -1, 1, 0);
            }
        }
        cv::cvtColor(image_lidar_pts, image_lidar_pts, CV_BGR2GRAY);
//        findSimilarity(edge_points_lidar, image_edge_gray);

        cv::imshow("image_lidar_pts", image_lidar_pts);
        cv::waitKey(1);

        cv::imshow("projected image", image_out);
        cv::waitKey(1);
        correlation(image_lidar_pts, image_edge_gray);
    }

    void callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                  const sensor_msgs::ImageConstPtr &image_msg) {
        double time1 = cloud_msg->header.stamp.toSec();
        double time2 = image_msg->header.stamp.toSec();
        double time_diff = time1 - time2;
        if(fabs(time_diff) <= 0.01) {
//            ROS_INFO_STREAM("Time diff: " << time_diff);
            lidar_frameId = cloud_msg->header.frame_id;
            objectPoints_L.clear();
            objectPoints_C.clear();
            imagePoints.clear();
            publishTransforms();
            image_in = cv_bridge::toCvShare(image_msg, "mono8")->image;
            pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*cloud_msg, *in_cloud);

            cv::cvtColor(image_in, image_out, CV_GRAY2BGR);
            image_projected = cv::Mat::zeros(image_height, image_width, CV_8UC3);
            double fov_x, fov_y;
            fov_x = 2*atan2(image_width, 2*projection_matrix.at<double>(0, 0))*180/CV_PI;
            fov_y = 2*atan2(image_height, 2*projection_matrix.at<double>(1, 1))*180/CV_PI;

            double max_range, min_range;
            double min_height, max_height;
            max_range = min_height = -INFINITY;
            min_range = max_height = INFINITY;

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
                pointCloud_C = C_T_L.block(0, 0, 3, 4)*pointCloud_L;

                double X = pointCloud_C[0];
                double Y = pointCloud_C[1];
                double Z = pointCloud_C[2];

                double Xangle = atan2(X, Z)*180/CV_PI;
                double Yangle = atan2(Y, Z)*180/CV_PI;

                if(Xangle < -fov_x/2 || Xangle > fov_x/2)
                    continue;

                if(Yangle < -fov_y/2 || Yangle > fov_y/2)
                    continue;

                double range = Z;
                if(range > max_range) {
                    max_range = range;
                }

                if(range < min_range) {
                    min_range = range;
                }

                if(Z > max_height)
                    max_height = Z;

                if(Z < min_height)
                    min_height = Z;

                objectPoints_L.push_back(cv::Point3d(pointCloud_L[0], pointCloud_L[1], pointCloud_L[2]));
                objectPoints_C.push_back(cv::Point3d(X, Y, Z));

                cv::projectPoints(objectPoints_L, rvec, tvec, projection_matrix, distCoeff, imagePoints, cv::noArray());
            }

            /// Color the Point Cloud
            colorPointCloud();

            pcl::toROSMsg(out_cloud_pcl, out_cloud_ros);
            out_cloud_ros.header.frame_id = cloud_msg->header.frame_id;
            out_cloud_ros.header.stamp = cloud_msg->header.stamp;

            cloud_pub.publish(out_cloud_ros);

            /// Color Lidar Points on the image a/c to distance
            colorLidarPointsOnImage(min_range, max_range, min_height, max_height);
            sensor_msgs::ImagePtr msg =
                    cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_out).toImageMsg();
            image_pub.publish(msg);
            frame_no = image_msg->header.seq;
            ROS_WARN_STREAM("Frame no: " << frame_count);
            if(++frame_count == 50) {
                ROS_INFO_STREAM("No of data frames collected: " << dataFrames.size());
                ros::shutdown();
            }
        } else {
            ROS_WARN_STREAM("Time Diff too high!");
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "cam_lidar_proj");
    lidarImageProjection lip;
    ros::spin();
    return 0;
}