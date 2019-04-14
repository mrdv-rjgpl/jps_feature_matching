#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include <string>

#include <algorithm>
#include <functional>

#include <exception>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class FeatureMatcher
{
  private:
    image_transport::ImageTransport *img_transport;
    image_transport::Subscriber image_sub;
    int min_hessian;
    Mat img_template;
    Mat desc_template;
    ros::NodeHandle nh;
    ros::Publisher image_pub;
    vector<KeyPoint> kp_template;
    Ptr<SURF> surf_detector;
    double dist_ratio_pdf[21][2];

  public:
    FeatureMatcher(ros::NodeHandle& nh, int min_hessian);
    void imageSubscriberCallback(const sensor_msgs::ImageConstPtr& msg);
};

FeatureMatcher::FeatureMatcher(ros::NodeHandle& nh, int min_hessian)
{
  string img_template_name;

  ROS_INFO("Initializing feature matcher...");
  this->nh = nh;
  this->min_hessian = min_hessian;
  this->surf_detector = SURF::create(min_hessian);
  this->img_transport = new image_transport::ImageTransport(this->nh);
  this->image_sub = this->img_transport->subscribe(
      "/piece_parser/image_piece",
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
    this->surf_detector->detectAndCompute(this->img_template, noArray(), kp_template, desc_template);
  }
  else
  {
    ROS_ERROR("Failed to get parameter 'img_template_name'.");
  }
}

void FeatureMatcher::imageSubscriberCallback(const sensor_msgs::ImageConstPtr& msg)
{
  ROS_INFO("Extracting SURF features from puzzle piece...");
  Mat img_input = cv_bridge::toCvShare(msg, "bgr8")->image;
  vector<KeyPoint> kp_piece;
  Mat desc_piece;

  // Extract SURF features from input image of puzzle piece.
  this->surf_detector->detectAndCompute(
      cv_bridge::toCvShare(msg, "bgr8")->image,
      noArray(),
      kp_piece,
      desc_piece);
  // Compare with SURF features extracted from template image.
}

//double FeatureMatcher::computeSurfScore(vector<KeyPoint> kp_input)
//{
//  double score = 0.0;
//  int i;
//
//  for(i = 0; i < kp_input.size(); ++i)
//  {
//    score += 1.0 / (1.0);
//  }
//}

int main(int argc, char **argv)
{
  return 0;
}

