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
#include <geometry_msgs/Point.h>

#include <string>

#include <algorithm>
#include <functional>
#include <exception>

#include "jps_feature_matching/ImageTransform.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class MatchedFeatureViz
{
  private:
    /*
     * \brief Node handler object
     */
    ros::NodeHandle nh;
    /*
     * \brief Image subscriber object
     */
    ros::Subscriber image_sub;
    /*
     * \brief Image transport object
     */
    image_transport::ImageTransport *img_transport;
    /*
     * \brief Image publisher object
     */
    image_transport::Publisher image_pub;

  public:
    MatchedFeatureViz(ros::NodeHandle& nh);
    void imageSubscriberCallback(
        const jps_feature_matching::ImageTransformConstPtr& msg);
};

MatchedFeatureViz::MatchedFeatureViz(ros::NodeHandle& nh)
{
  this->nh = nh;
  this->img_transport = new image_transport::ImageTransport(this->nh);
  this->image_sub = this->nh.subscribe(
      "input_image",
      1,
      &MatchedFeatureViz::imageSubscriberCallback,
      this);
  this->image_pub = this->img_transport->advertise("output_image", 1000);
}

void MatchedFeatureViz::imageSubscriberCallback(
    const jps_feature_matching::ImageTransformConstPtr& msg)
{
  int i;
  Mat img_input = cv_bridge::toCvCopy(msg->image, "bgr8")->image;
  vector<Point2f> central_points;
  sensor_msgs::ImagePtr vis_msg;

  for(i = 0; i < msg->transformed_points.size(); ++i)
  {
    central_points.push_back(Point2f(
          msg->transformed_points[i].x,
          msg->transformed_points[i].y));
    circle(
        img_input,
        central_points[i],
        16,
        Scalar(int(0.741 * 256), int(0.447 * 256), int(0.000 * 256)),
        -1,
        8,
        0);
  }

  vis_msg = cv_bridge::CvImage(
      std_msgs::Header(), "bgr8", img_input).toImageMsg();
  this->image_pub.publish(vis_msg);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "matched_feature_viz");
  ros::NodeHandle nh;
  MatchedFeatureViz m(nh);
  ros::spin();
  return 0;
}

