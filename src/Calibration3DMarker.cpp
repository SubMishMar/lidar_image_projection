/*
 * Calibration3DMarker.cpp
 *
 *  Created on: 2.4.2014
 *      Author: ivelas
 */

#include "Calibration3DMarker.h"

#include <ros/assert.h>
#include <pcl/visualization/cloud_viewer.h>
using namespace std;
using namespace cv;
using namespace pcl;
using namespace ros;



Calibration3DMarker::Calibration3DMarker(cv::Mat _frame_gray, cv::Mat _P, ::PointCloud<Velodyne::Point> _pc,
                                         float _circ_distance, float _radius) :
    frame_gray(_frame_gray), P(_P), pc(_pc), circ_distance(_circ_distance), radius(_radius)
{

  // ---------------- GET PLANE ----------------

  Velodyne::Velodyne scan(pc);
  scan.getRings();
  scan.intensityByRangeDiff();
  PointCloud<pcl::PointXYZ>::Ptr cloud4viewing(scan.toPointsXYZ());
  //Velodyne::Velodyne::view(cloud4viewing);

  PointCloud<Velodyne::Point> visible_cloud;
  scan.project(P, Rect(0, 0, 2048, 2592), &visible_cloud);

  Velodyne::Velodyne visible_scan(visible_cloud);
  Velodyne::Velodyne::view(PointCloud<PointXYZ>::Ptr(visible_scan.toPointsXYZ()));
  visible_scan.normalizeIntensity();
  Velodyne::Velodyne thresholded_scan = visible_scan.threshold(0.1);

  PointCloud<PointXYZ>::Ptr xyz_cloud_ptr(thresholded_scan.toPointsXYZ());
  //Velodyne::Velodyne::view(xyz_cloud_ptr);

  SampleConsensusModelPlane<PointXYZ>::Ptr model_p(
      new ::SampleConsensusModelPlane<PointXYZ>(xyz_cloud_ptr));
  RandomSampleConsensus<PointXYZ> ransac(model_p);
  ransac.setDistanceThreshold(0.05);
  ransac.computeModel();

  std::vector<int> inliers_indicies;
  ransac.getInliers(inliers_indicies);

  copyPointCloud<PointXYZ>(*xyz_cloud_ptr, inliers_indicies, plane);

  // ---------------- REMOVE LINES ----------------

  for (int i = 0; i < 2; i++)
  {
    PointCloud<PointXYZ>::Ptr plane_ptr(new PointCloud<PointXYZ>(plane));
    SampleConsensusModelLine<PointXYZ>::Ptr model_l(
        new SampleConsensusModelLine<PointXYZ>(plane_ptr));
    RandomSampleConsensus<PointXYZ> ransac_l(model_l);
    ransac_l.setDistanceThreshold(0.02);
    ransac_l.computeModel();
    vector<int> line_inliers;
    ransac_l.getInliers(line_inliers);
    if (line_inliers.empty())
    {
      continue;
    }
    PointCloud<PointXYZ> plane_no_line;
    remove_inliers(*plane_ptr, line_inliers, plane_no_line);
    plane = plane_no_line;
  }

}

bool Calibration3DMarker::detectCirclesInImage(vector<Point2f> &centers, vector<float> &radiuses)
{
  Image::Image img(frame_gray);
  Image::Image img_edge(img.computeEdgeImage());

  cv::GaussianBlur( frame_gray, frame_gray, cv::Size(9, 9), 0.5, 0.5 );
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles( frame_gray, circles, CV_HOUGH_GRADIENT, 1, frame_gray.rows/8, 200, 100, 0, 0 );

  for (size_t i = 0; i < circles.size(); i++) {
      centers.push_back(Point2f(circles[i][0], circles[i][1]));
      radiuses.push_back(cvRound(circles[i][2]));
  }
    /// Draw the circles detected

//     Mat src_rgb;
//     Mat src_rgb;
//     cvtColor(frame_gray, src_rgb, CV_GRAY2BGR );
//     vector<Scalar> colors;
//     colors.push_back(Scalar(255,0,0));
//     colors.push_back(Scalar(0,255,0));
//     colors.push_back(Scalar(0,0,255));
//     colors.push_back(Scalar(255,255,255));
//     for (size_t i = 0; i < circles.size(); i++) {
//     centers.push_back(Point2f(circles[i][0], circles[i][1]));
//
//     Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
//     int radius = cvRound(circles[i][2]);
//
//     // circle center
//     circle(src_rgb, center, 3, Scalar(0, 255, 0), -1, 8, 0);
//     // circle outline
//     circle(src_rgb, center, radius, colors[i], 3, 8, 0);
//     cerr << i+1 << ". circle S("<<center.x<<","<<center.y<<"); r="<<radius << endl;
//     }
//     Mat frame_downsized;
//     cv::resize(src_rgb, frame_downsized, cv::Size(), 0.5, 0.5);
//     namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
//     imshow("Hough Circle Transform Demo", frame_downsized);
//     waitKey(0);


  if (circles.size() != 4) {
      ROS_WARN_STREAM("No of circles detected is not 4");
        return false;
  } else {
      ROS_INFO_STREAM("No of circles detected is 4");
      return true;
  }
}

bool Calibration3DMarker::detectCirclesInPointCloud(vector<Point3f> &centers, vector<float> &radiuses)
{
  PointCloud<PointXYZ>::Ptr detection_cloud(new PointCloud<PointXYZ>);
  *detection_cloud += this->plane;

  float tolerance = 0.03; // 3cm
  int round = 1;
  vector<PointXYZ> spheres_centers;
  bool detected = false;
  for (int iterations = 0; iterations < 64; iterations++)
  {
    /*cerr << endl << " =========== ROUND " << round++ << " =========== "
     << endl << endl;
     cerr << "detection_cloud size: " << detection_cloud->size() << endl;*/
    spheres_centers = detect4spheres(detection_cloud, radiuses);

    if (spheres_centers.size() == 4)
    {
      order4spheres(spheres_centers);
      /*cerr << "ordered centers: " << endl;
       for (size_t i = 0; i < spheres_centers.size(); i++) {
       cerr << spheres_centers[i] << endl;
       }*/
      if (verify4spheres(spheres_centers, this->circ_distance, tolerance))
      {
        spheres_centers = refine4centers(spheres_centers, detection_cloud);
        detected = true;
        break;
      }
    }
    vector<PointXYZ> possible_centers = generate_possible_centers(spheres_centers, this->circ_distance);
    generate_possible_points(this->plane, detection_cloud, possible_centers, this->circ_distance, 0.01);
  }

  if (!detected)
  {
    return false;
  }

  for (size_t i = 0; i < spheres_centers.size(); i++)
  {
    centers.push_back(Point3f(spheres_centers[i].x, spheres_centers[i].y, spheres_centers[i].z));
  }
  return true;
}

vector<PointXYZ> Calibration3DMarker::detect4spheres(PointCloud<PointXYZ>::Ptr plane,
                                                          vector<float> &radiuses)
{

  radiuses.clear();
  vector<PointXYZ> centers;
  std::vector<int> inliers_indicies;
  PointCloud<PointXYZ> *four_spheres = new PointCloud<PointXYZ>();
  float tolerance = 0.02;
  for (int i = 0; i < 4; i++)
  {
    SampleConsensusModelSphere<PointXYZ>::Ptr model_s(
        new SampleConsensusModelSphere<PointXYZ>(plane));
    model_s->setRadiusLimits(0.08, 0.09);
    RandomSampleConsensus<PointXYZ> ransac_sphere(model_s);
    ransac_sphere.setDistanceThreshold(tolerance);
    ransac_sphere.computeModel();
    inliers_indicies.clear();
    ransac_sphere.getInliers(inliers_indicies);

    if (inliers_indicies.size() == 0)
    {
      continue;
    }
    Eigen::VectorXf coeficients;
    ransac_sphere.getModelCoefficients(coeficients);
    //cerr << i + 1 << ". circle: " << coeficients << endl << endl;

    PointCloud<PointXYZ>::Ptr outliers(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr inliers(new PointCloud<PointXYZ>);
    remove_inliers<PointXYZ>(*plane, inliers_indicies, *outliers);
    copyPointCloud<PointXYZ>(*plane, inliers_indicies, *inliers);
    plane = outliers;
    //view(plane);

    *four_spheres += *inliers;
    PointXYZ middle(coeficients(0), coeficients(1), coeficients(2));
    four_spheres->push_back(middle);
    centers.push_back(middle);

    float radius = coeficients(3);
    radiuses.push_back(radius);
  }
  PointCloud<PointXYZ>::Ptr four_spheres_ptr(four_spheres);
  return centers;
}

/*
 * Indexes of circles in marker:
 *
 * 0 1
 * 2 3
 */
bool orderX(PointXYZ p1, PointXYZ p2)
{
  return p1.x < p2.x;
}
bool orderY(PointXYZ p1, PointXYZ p2)
{
  return p1.y < p2.y;
}

void Calibration3DMarker::order4spheres(vector<PointXYZ> &spheres_centers)
{
  ROS_ASSERT(spheres_centers.size() == 4);
  sort(spheres_centers.begin(), spheres_centers.end(), orderY);
  sort(spheres_centers.begin(), spheres_centers.begin() + 2, orderX);
  sort(spheres_centers.begin() + 2, spheres_centers.begin() + 4, orderX);
}

float euclid_dist(const PointXYZ p1, const PointXYZ p2)
{
  return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
}

bool Calibration3DMarker::verify4spheres(const vector<PointXYZ> &spheres_centers, float straight_distance,
                                         float delta)
{
  ROS_ASSERT(spheres_centers.size() == 4);

  vector<pair<int, int> > neighbour_indexes;
  neighbour_indexes.push_back(pair<int, int>(0, 1));
  neighbour_indexes.push_back(pair<int, int>(1, 3));
  neighbour_indexes.push_back(pair<int, int>(3, 2));
  neighbour_indexes.push_back(pair<int, int>(2, 0));

  bool res = true;
  for (vector<pair<int, int> >::iterator neighbors = neighbour_indexes.begin(); neighbors < neighbour_indexes.end();
      neighbors++)
  {
    float error = abs(
        euclid_dist(spheres_centers[neighbors->first], spheres_centers[neighbors->second]) - straight_distance);
    //cerr << "error: " << error << endl;
    if (error > delta)
      res = false;
  }
  return res;
}

/*
 * All points around the all found centers:
 * x x x
 * x   x
 * x x x
 */
vector<PointXYZ> Calibration3DMarker::generate_possible_centers(const vector<PointXYZ> &spheres_centers,
                                                                     float straight_distance)
{
  vector<PointXYZ> possible_centers;

  for (vector<PointXYZ>::const_iterator c = spheres_centers.begin(); c < spheres_centers.end(); c++)
  {
    for (int dx = -1; dx <= 1; dx++)
    {
      for (int dy = -1; dy <= 1; dy++)
      {
        if (dx || dy)
        { // omitting found center (may be false detection)
          PointXYZ new_center = *c;
          new_center.x += dx * straight_distance;
          new_center.y += dy * straight_distance;
          possible_centers.push_back(new_center);
        }
      }
    }
  }

  return possible_centers;
}

void Calibration3DMarker::generate_possible_points(PointCloud<PointXYZ> &plane,
                                                   PointCloud<PointXYZ>::Ptr detection_cloud,
                                                   const vector<PointXYZ> &possible_centers, float radius,
                                                   float tolerance)
{

  detection_cloud->clear();
  for (PointCloud<PointXYZ>::iterator pt = plane.begin(); pt < plane.end(); pt++)
  {
    int votes = 0;
    for (vector<PointXYZ>::const_iterator center = possible_centers.begin(); center < possible_centers.end();
        center++)
    {
      if (euclid_dist(*pt, *center) < radius + tolerance)
      {
        votes++;
      }
    }
    if (votes > 0)
    {
      detection_cloud->push_back(*pt);
    }
  }
}

vector<PointXYZ> Calibration3DMarker::refine4centers(vector<PointXYZ> centers,
                                                     PointCloud<PointXYZ>::Ptr detection_cloud)
{

  float z_coord = 0;
  for (PointCloud<PointXYZ>::iterator pt = detection_cloud->begin(); pt < detection_cloud->end(); pt++)
  {
    z_coord += pt->z;
  }
  z_coord /= detection_cloud->size();

  for (vector<PointXYZ>::iterator c = centers.begin(); c < centers.end(); c++)
  {
    c->z = z_coord;
  }

  return centers;
}


