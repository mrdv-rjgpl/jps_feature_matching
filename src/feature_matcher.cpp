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
     * \brief Image transport object
     */
    image_transport::ImageTransport *img_transport;
    /*
     * \brief Minimum hessian threshold for SURF feature extraction
     */
    int min_hessian;
    /*
     * \brief Template image
     */
    Mat img_template;
    /*
     * \brief Piece template
     */
    Mat piece_template;
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
    image_transport::Publisher image_pub;
    /*
     * \brief SURF feature matcher
     */
    Ptr<DescriptorMatcher> matcher;
    /*
     * \brief Keypoint vector for template image
     */
    vector<KeyPoint> kp_template;
    /*
     * \brief List of piece indices for each key point in the template image
     */
    vector<int> kp_piece_indices;
    vector< vector<Point2f> > piece_contours_f;
    /*
     * \brief SURF detector object
     */
    Ptr<SURF> surf_detector;
    /*
     * \brief Extract piece positions from the piece template
     */
    void getPieceTemplate(string piece_template_name);

  public:
    /*
     * \brief Constructor for FeatureMatcher class
     *
     * \param[in] nh The node handler object
     * \param[in] min_hessian The minimum Hessian threshold for SURF
     * feature extraction
     *
     * \return a FeatureMatcher object
     */
    FeatureMatcher(ros::NodeHandle& nh, int min_hessian);

    /*
     * \brief Callback function for image subscriber
     *
     * \param[in] msg A message of type ImageWithContour containing the
     * image, the contour of the piece, and its centroid
     */
    void imageSubscriberCallback(
        const jps_puzzle_piece::ImageWithContourPtr& msg);
};

FeatureMatcher::FeatureMatcher(ros::NodeHandle& nh, int min_hessian)
{
  int i;
  int j;
  int k;
  Mat contour_display;
  Mat piece_template;
  Mat piece_template_bin;
  Mat piece_template_gray;
  Mat piece_template_temp;
  string img_template_name;
  string piece_template_name;
  vector< vector<Point> > piece_contours;
  vector<Vec4i> contour_heirarchy;

  // Initialize the feature matching node variables.
  ROS_INFO("Initializing feature matching node...");
  this->nh = nh;
  this->img_transport = new image_transport::ImageTransport(this->nh);
  this->min_hessian = min_hessian;
  this->surf_detector = SURF::create(min_hessian);
  this->image_sub = this->nh.subscribe(
      "input_image",
      1,
      &FeatureMatcher::imageSubscriberCallback,
      this);
  this->image_pub = this->img_transport->advertise("output_image", 1000);

  if(this->nh.getParam("img_template_name", img_template_name))
  {
    // Load the template image containing the solved puzzle.
    ROS_INFO_STREAM("Loading template image " << img_template_name << "...");
    this->img_template = imread(img_template_name, CV_LOAD_IMAGE_COLOR);
    // Extract and store SURF features from the template image.
    ROS_INFO_STREAM("Extracting SURF features from template image...");
    this->surf_detector->detectAndCompute(
        this->img_template,
        noArray(),
        this->kp_template,
        this->desc_template);
  }
  else
  {
    ROS_ERROR("Failed to get parameter 'img_template_name'.");
  }

  if(this->nh.getParam("piece_template_name", piece_template_name))
  {
    // Load the template image containing the contours of the puzzle pieces.
    ROS_INFO_STREAM("Loading piece template " << piece_template_name << "...");
    piece_template_temp = imread(piece_template_name, CV_LOAD_IMAGE_COLOR);

    // Resize the piece template to match the image template.
    ROS_INFO_STREAM(
        "Resizing piece template ("
        << piece_template_temp.rows << "x" << piece_template_temp.cols
        << ") to match image template ("
        << this->img_template.rows << "x" << this->img_template.cols
        << ")...");
    resize(piece_template_temp, piece_template, this->img_template.size());

    // Grayscale and binarize the piece template with a hard coded threshold.
    ROS_INFO("Grayscaling and binarizing piece template...");
    piece_template_gray = Mat(piece_template.size(), CV_8UC1);
    cvtColor(piece_template, piece_template_gray, CV_RGB2GRAY);
    piece_template_bin = Mat(piece_template.size(), CV_8UC1);
    threshold(
        piece_template_gray, piece_template_bin, 100, 255, THRESH_BINARY_INV);

    // Find the piece contours in the piece template.
    ROS_INFO("Finding piece contours in piece template...");
    findContours(
        piece_template_bin,
        piece_contours,
        contour_heirarchy,
        CV_RETR_TREE,
        CV_CHAIN_APPROX_NONE);
    contour_display = Mat::zeros(piece_template.size(), CV_8UC3);
    Scalar colour(int(0.741 * 255), int(0.447 * 255), int(0.000 * 255));

    // Prune the outermost border from the piece contour list.
    for(i = 0; i < contour_heirarchy.size(); ++i)
    {
      if(contour_heirarchy[i][3] >= 0)
      {
        // Add the current piece contour to piece_contours_f.
        this->piece_contours_f.push_back(vector<Point2f>());
        k = this->piece_contours_f.size() - 1;

        for(j = 0; j < piece_contours[i].size(); ++j)
        {
          this->piece_contours_f[k].push_back(
              Point2f(piece_contours[i][j].x, piece_contours[i][j].y));
        }
      }
      else
      {
        // No operation
      }
    }

    // Catalog the SURF features according to their puzzle piece location.
    for(i = 0; i < this->kp_template.size(); ++i)
    {
      for(j = 0; j < this->piece_contours_f.size(); ++j)
      {
        if(pointPolygonTest(
              this->piece_contours_f[j],
              kp_template[i].pt,
              false) > 0)
        {
          ROS_INFO_STREAM("SURF feature " << i << " is in piece " << j << ".");
          this->kp_piece_indices.push_back(j);
          break;
        }
        else
        {
          // No operation
        }
      }
    }
  }
  else
  {
    ROS_ERROR("Failed to get parameter 'piece_template_name'.");
  }

  ROS_INFO("Instantiating SURF feature matcher...");
  this->matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  ROS_INFO("Waiting for messages on topic 'input_image'...");
}

void FeatureMatcher::imageSubscriberCallback(
    const jps_puzzle_piece::ImageWithContourPtr& msg)
{
  double dist_ratio_threshold = 0.7;
  int i;
  int j;
  int likely_piece_index = 0;
  Mat desc_piece = cv_bridge::toCvCopy(
      msg->surf_desc,
      sensor_msgs::image_encodings::TYPE_32FC1)->image;
  Mat h;
  sensor_msgs::ImagePtr image_pub_msg;
  vector< vector<DMatch> > knn_matches;
  // Good SURF matches, binned according to the piece in the template.
  vector< vector<DMatch> > good_matches(this->piece_contours_f.size());
  // SURF feature points in the puzzle piece,
  // binned according to the template piece they are matched to.
  vector< vector<Point2f> > pt_piece(this->piece_contours_f.size());
  // SURF feature points from the template,
  // binned according the piece they lie in.
  vector< vector<Point2f> > pt_template(this->piece_contours_f.size());

  // Compare with SURF features extracted from template image.
  // https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
  ROS_INFO_STREAM("Matching SURF features from image with template...");
  this->matcher->knnMatch(desc_piece, this->desc_template, knn_matches, 2);
  ROS_INFO("Running comparison for good matches...");

  for(i = 0; i < knn_matches.size(); ++i)
  {
    if((knn_matches[i][0].distance
          < dist_ratio_threshold * knn_matches[i][1].distance))
    {
      // Determine which piece in the template this match throws.
      j = this->kp_piece_indices[knn_matches[i][0].trainIdx];
      // Populate the good matches, piece points,
      // and template points accordingly.
      good_matches[j].push_back(knn_matches[i][0]);
      pt_piece[j].push_back(Point2f(
            msg->surf_key_points[knn_matches[i][0].queryIdx].x,
            msg->surf_key_points[knn_matches[i][0].queryIdx].y));
      pt_template[j].push_back(
          this->kp_template[knn_matches[i][0].trainIdx].pt);
    }
    else
    {
      // No operation
    }
  }

  // Determine the index of the most likely piece to which this corresponds.
  for(i = 0; i < good_matches.size(); ++i)
  {
    if(good_matches[i].size() > good_matches[likely_piece_index].size())
    {
      likely_piece_index = i;
    }
    else
    {
      // No operation
    }
  }

  // Compute the homographic transformation.
  ROS_INFO(
      "Computing homographic transformation from piece to template...");
  h = findHomography(
      pt_piece[likely_piece_index],
      pt_template[likely_piece_index],
      CV_RANSAC);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "feature_matcher_node");
  ros::NodeHandle nh;
  FeatureMatcher p(nh, 400.0);
  ros::spin();

  return 0;
}

