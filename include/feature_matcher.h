#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

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
#include <geometry_msgs/PoseStamped.h>

#include <string>

#include <algorithm>
#include <functional>
#include <exception>

#include "jps_puzzle_piece/ImageWithContour.h"
#include "jps_feature_matching/FindPieceTransform.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class FeatureMatcher
{
  private:
    bool robot_stationary;
    /*
     * \brief Transform publisher object
     */
    ros::Publisher transform_pub;
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
    Mat img_piece;
    /*
     * \brief Piece template
     */
    Mat piece_template;
    /*
     * \brief Descriptor matrix for template image
     */
    Mat desc_template;
    Mat desc_piece;
    /*
     * \brief Node handler object
     */
    ros::NodeHandle nh;
    /*
     * \brief Image publisher object
     */
    image_transport::Publisher vis_pub;
    /*
     * \brief SURF feature matcher
     */
    Ptr<DescriptorMatcher> matcher;
    /*
     * \brief SURF detector object
     */
    Ptr<SURF> surf_detector;
    /*
     * \brief Keypoint vector for template image
     */
    vector<KeyPoint> kp_template;
    /*
     * \brief List of piece indices for each key point in the template image
     */
    vector<int> kp_piece_indices;
    /*
     * \brief Contours of each piece extracted from the template in floating point form
     */
    vector< vector<Point2f> > piece_contours_f;
    /*
     * \brief Contours of each piece extracted from the template in integer form
     */
    vector< vector<Point> > piece_contours_i;
    /*
     * \brief Centroid and two other points to form x-y axes for each piece
     */
    vector< vector<Point2f> > piece_central_points;
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

    bool findPieceTransform(
        jps_feature_matching::FindPieceTransform::Request &req,
        jps_feature_matching::FindPieceTransform::Response &rsp);

    /*
     * \brief Callback function for image subscriber
     *
     * \param[in] msg A message of type ImageWithContour containing the
     * image, the contour of the piece, and its centroid
     */
    void imageSubscriberCallback(
        const jps_puzzle_piece::ImageWithContourPtr& msg);
};

geometry_msgs::Quaternion AxesToQuaternion(Point3f x_axis, Point3f y_axis, Point3f z_axis);
#endif /* FEATURE_MATCHER_H */
