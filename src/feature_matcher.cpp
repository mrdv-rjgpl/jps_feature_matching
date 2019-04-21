#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include <string>

#include <algorithm>
#include <functional>

#include <exception>
#include "jps_puzzle_piece/ImageWithContour.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class FeatureMatcher
{
  private:
    /*
     * \brief Image subscriber object
     */
    ros::Subscriber image_sub;
    /*
     * \brief Minimum hessian threshold for SURF feature extraction
     */
    int min_hessian;
    /*
     * \brief Template image
     */
    Mat img_template;
    /*
     * \brief Descriptor matrix for template image
     */
    Mat desc_template;
    /*
     * \brief Node handler object
     */
    ros::NodeHandle nh;
    /*
     * \brief Image publisher object
     */
    ros::Publisher image_pub;
    /*
     * \brief Keypoint vector for template image
     */
    vector<KeyPoint> kp_template;
    /*
     * \brief SURF detector object
     */
    Ptr<SURF> surf_detector;
    /*
     * \brief Probability Distribution Function of validity
     *
     * \details The validity PDF is computed with respect to the ratio
     * of the smallest to the second smallest distances.
     */
    double dist_ratio_pdf[21][2];

  public:
    /*
     * \brief Constructor for FeatureMatcher class
     *
     * \param[in] nh The node handler object
     * \param[in] min_hessian The minimum Hessian threshold for SURF feature
     * extraction
     *
     * \return a FeatureMatcher object
     */
    FeatureMatcher(ros::NodeHandle& nh, int min_hessian);

    /*
     * \brief Callback function for image subscriber
     *
     * \param[in] msg A message of type ImageWithContour containing the image,
     * the contour of the piece, and its centroid
     */
    void imageSubscriberCallback(
        const jps_puzzle_piece::ImageWithContourPtr& msg);
};

FeatureMatcher::FeatureMatcher(ros::NodeHandle& nh, int min_hessian)
{
  string img_template_name;

  ROS_INFO("Initializing feature matcher...");
  this->nh = nh;
  this->min_hessian = min_hessian;
  this->surf_detector = SURF::create(min_hessian);
  this->image_sub = this->nh.subscribe(
      "input_image",
      1,
      &FeatureMatcher::imageSubscriberCallback,
      this);

  this->dist_ratio_pdf[ 0][0] = 0.00;
  this->dist_ratio_pdf[ 0][1] = 0.000;
  this->dist_ratio_pdf[ 1][0] = 0.05;
  this->dist_ratio_pdf[ 1][1] = 0.000;
  this->dist_ratio_pdf[ 2][0] = 0.10;
  this->dist_ratio_pdf[ 2][1] = 0.000;
  this->dist_ratio_pdf[ 3][0] = 0.15;
  this->dist_ratio_pdf[ 3][1] = 0.000;
  this->dist_ratio_pdf[ 4][0] = 0.20;
  this->dist_ratio_pdf[ 4][1] = 0.025;
  this->dist_ratio_pdf[ 5][0] = 0.25;
  this->dist_ratio_pdf[ 5][1] = 0.050;
  this->dist_ratio_pdf[ 6][0] = 0.30;
  this->dist_ratio_pdf[ 6][1] = 0.150;
  this->dist_ratio_pdf[ 7][0] = 0.35;
  this->dist_ratio_pdf[ 7][1] = 0.250;
  this->dist_ratio_pdf[ 8][0] = 0.40;
  this->dist_ratio_pdf[ 8][1] = 0.275;
  this->dist_ratio_pdf[ 9][0] = 0.45;
  this->dist_ratio_pdf[ 9][1] = 0.300;
  this->dist_ratio_pdf[10][0] = 0.50;
  this->dist_ratio_pdf[10][1] = 0.250;
  this->dist_ratio_pdf[11][0] = 0.55;
  this->dist_ratio_pdf[11][1] = 0.200;
  this->dist_ratio_pdf[12][0] = 0.60;
  this->dist_ratio_pdf[12][1] = 0.155;
  this->dist_ratio_pdf[13][0] = 0.65;
  this->dist_ratio_pdf[13][1] = 0.110;
  this->dist_ratio_pdf[14][0] = 0.70;
  this->dist_ratio_pdf[14][1] = 0.085;
  this->dist_ratio_pdf[15][0] = 0.75;
  this->dist_ratio_pdf[15][1] = 0.060;
  this->dist_ratio_pdf[16][0] = 0.80;
  this->dist_ratio_pdf[16][1] = 0.045;
  this->dist_ratio_pdf[17][0] = 0.85;
  this->dist_ratio_pdf[17][1] = 0.030;
  this->dist_ratio_pdf[18][0] = 0.90;
  this->dist_ratio_pdf[18][1] = 0.020;
  this->dist_ratio_pdf[19][0] = 0.95;
  this->dist_ratio_pdf[19][1] = 0.010;
  this->dist_ratio_pdf[20][0] = 1.00;
  this->dist_ratio_pdf[20][1] = 0.000;

  if(this->nh.getParam("img_template_name", img_template_name))
  {
    ROS_INFO_STREAM("Loading template image " << img_template_name << "...");
    this->img_template = imread(img_template_name, CV_LOAD_IMAGE_COLOR);
    ROS_INFO("Extracting SURF features from template image...");
    this->surf_detector->detectAndCompute(
        this->img_template,
        noArray(),
        this->kp_template,
        this->desc_template);
    ROS_INFO_STREAM(
        this->kp_template.size()
        << " template elements extracted.");
  }
  else
  {
    ROS_ERROR("Failed to get parameter 'img_template_name'.");
  }
}

void FeatureMatcher::imageSubscriberCallback(
    const jps_puzzle_piece::ImageWithContourPtr& msg)
{
  double dist_ratio_threshold = 0.7;
  int i;
  Mat desc_piece;
  Mat h;
  Mat img_input = cv_bridge::toCvCopy(msg->image, "bgr8")->image;
  Mat img_matches;
  Mat img_template_cp;
  Ptr<DescriptorMatcher> matcher;
  Scalar colour(int(0.741 * 256), int(0.447 * 256), int(0.000 * 256));
  vector<DMatch> good_matches;
  vector<KeyPoint> kp_piece;
  vector<Point2f> pt_piece;
  vector<Point2f> pt_template;
  vector< vector<DMatch> > knn_matches;
  vector< vector<Point> > pt_piece_contours;
  vector< vector<Point> > pt_template_contours;
  vector< vector<Point2f> > pt_piece_contours_f;
  vector< vector<Point2f> > pt_template_contours_f;

  // Extract SURF features from input image of puzzle piece.
  ROS_INFO("Extracting SURF features from puzzle piece...");
  this->surf_detector->detectAndCompute(
      cv_bridge::toCvCopy(msg->image, "bgr8")->image,
      noArray(),
      kp_piece,
      desc_piece);
  ROS_INFO_STREAM(
      kp_piece.size()
      << " template elements extracted from puzzle piece.");

  // Compare with SURF features extracted from template image.
  // https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
  matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  ROS_INFO("Running k-NN matching...");
  matcher->knnMatch(desc_piece, this->desc_template, knn_matches, 2);
  ROS_INFO("Running comparison for good matches...");
  pt_piece_contours.push_back(vector<Point>());
  pt_piece_contours_f.push_back(vector<Point2f>());

  for(i = 0; i < msg->contour_px.size(); ++i)
  {
    pt_piece_contours[0].push_back(
        Point(msg->contour_px[i].x, msg->contour_px[i].y));
    pt_piece_contours_f[0].push_back(
        Point2f(msg->contour_px[i].x, msg->contour_px[i].y));
  }

  ROS_INFO_STREAM(
      pt_piece_contours_f[0].size()
      << " contour points obtained for current piece.");

  for(i = 0; i < knn_matches.size(); ++i)
  {
    if((knn_matches[i][0].distance
          < dist_ratio_threshold * knn_matches[i][1].distance)
        && (pointPolygonTest(
            pt_piece_contours_f[0],
            kp_piece[knn_matches[i][0].queryIdx].pt,
            false)
          > 0))
    {
      good_matches.push_back(knn_matches[i][0]);
      pt_piece.push_back(kp_piece[knn_matches[i][0].queryIdx].pt);
      pt_template.push_back(
          this->kp_template[knn_matches[i][0].trainIdx].pt);
    }
    else
    {
      // No operation
    }
  }

  ROS_INFO_STREAM(good_matches.size() << " good SURF matches found.");
  ROS_INFO("Drawing piece centroid...");
  circle(
       img_input,
       Point2f(msg->centroid_px.x, msg->centroid_px.y),
       16,
       colour,
       -1,
       8,
       0);
  ROS_INFO("Drawing piece contour...");
  // Draw the piece contour with a line thickness of 2.
  drawContours(img_input, pt_piece_contours, 0, colour, 2);

  ROS_INFO(
      "Computing homographic transformation from piece to template...");
  h = findHomography(pt_piece, pt_template, CV_RANSAC);
  pt_template_contours_f.push_back(vector<Point2f>(
        pt_piece_contours_f[0].size()));

  ROS_INFO("Transforming piece contour into image template...");
  perspectiveTransform(pt_piece_contours_f[0], pt_template_contours_f[0], h);
  ROS_INFO("Transforming piece contour into image template...");
  img_template_cp = this->img_template.clone();
  pt_template_contours.push_back(vector<Point>());

  for(i = 0; i < pt_template_contours_f[0].size(); ++i)
  {
    pt_template_contours[0].push_back(
        Point(
          (int) pt_template_contours_f[0][i].x,
          (int) pt_template_contours_f[0][i].y));
  }

  drawContours(img_template_cp, pt_template_contours, 0, colour, 2);

  ROS_INFO("Drawing matched SURF features on piece and template...");
  drawMatches(
      img_input,
      kp_piece,
      img_template_cp,
      this->kp_template,
      good_matches,
      img_matches,
      Scalar::all(-1),
      Scalar::all(-1),
      vector<char>(),
      DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  imshow("Good matches", img_matches);
  waitKey(0);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "feature_matcher_node");
  ros::NodeHandle nh;
  FeatureMatcher p(nh, 400.0);
  ros::spin();

  return 0;
}

