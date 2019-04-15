#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include <string>

using namespace cv;
using namespace std;

class TestFeatureSupplier
{
  private:
    ros::NodeHandle nh;
    image_transport::ImageTransport *img_transport;
    image_transport::Publisher image_pub;
    ros::Timer timer;

  public:
    TestFeatureSupplier(ros::NodeHandle& nh);
    void timerCallback(const ros::TimerEvent& event);
};

TestFeatureSupplier::TestFeatureSupplier(ros::NodeHandle& nh)
{
  ROS_INFO("Initializing piece image supplier...");
  this->nh = ros::NodeHandle(nh);
  this->img_transport = new image_transport::ImageTransport(this->nh);
  this->image_pub = this->img_transport->advertise("/piece_parser/image_piece", 1);
  this->timer = this->nh.createTimer(ros::Duration(10.0), &TestFeatureSupplier::timerCallback, this, true);
}

void TestFeatureSupplier::timerCallback(const ros::TimerEvent& event)
{
  string img_file_name;
  ROS_INFO("Fetching image file name...");

  if(this->nh.getParam("img_file_name", img_file_name))
  {
    Mat img = imread(img_file_name, CV_LOAD_IMAGE_COLOR);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    ROS_INFO("Publishing image_transport message...");
    this->image_pub.publish(msg);
  }
  else
  {
    ROS_ERROR("Failed to get parameter 'img_file_name'.");
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_feature_supplier_node");
  ros::NodeHandle nh;
  TestFeatureSupplier p(nh);
  ros::spin();

  return 0;
}

