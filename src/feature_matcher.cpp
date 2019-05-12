#include "feature_matcher.h"

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
  this->vis_pub = this->img_transport->advertise("vis_image", 1000);
  this->transform_pub =
    this->nh.advertise<geometry_msgs::PoseStamped>(
        "homographic_transform",
        1000);
  this->robot_stationary = false;

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

    // Prune the outermost border from the piece contour list.
    for(i = 0; i < contour_heirarchy.size(); ++i)
    {
      if(contour_heirarchy[i][3] >= 0)
      {
        // Add the current piece contour to piece_contours_f.
        ROS_INFO_STREAM(
            "Adding current contour (" << i
            << ") to list of valid pieces...");
        this->piece_contours_f.push_back(vector<Point2f>());
        this->piece_contours_i.push_back(vector<Point>());
        k = this->piece_contours_f.size() - 1;

        for(j = 0; j < piece_contours[i].size(); ++j)
        {
          this->piece_contours_f[k].push_back(
              Point2f(piece_contours[i][j].x, piece_contours[i][j].y));
          this->piece_contours_i[k].push_back(
              Point(piece_contours[i][j].x, piece_contours[i][j].y));
        }

        // Find the centroid of the piece and two points to form the x-y axes.
        // ROS_INFO("Determining centroid of current piece...");
        Moments m = moments(piece_contours[i], false);
        // ROS_INFO("Expanding vector of centroid and x-y axis points...");
        this->piece_central_points.push_back(vector<Point2f>());
        // ROS_INFO("Saving centroid and x-y axis points for current piece...");
        this->piece_central_points[k].push_back(
            Point2f(m.m10 / m.m00, m.m01 / m.m00));
        this->piece_central_points[k].push_back(Point2f(
              this->piece_central_points[k][0].x + 32.0,
              this->piece_central_points[k][0].y));
        this->piece_central_points[k].push_back(Point2f(
              this->piece_central_points[k][0].x,
              this->piece_central_points[k][0].y + 32.0));
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
  Mat h_inv;
  Mat template_vis = this->img_template.clone();
  Mat img_piece = cv_bridge::toCvCopy(msg->image, "bgr8")->image;
  Mat img_vis;
  geometry_msgs::Point pt_temp;
  geometry_msgs::PoseStamped img_tf_msg;
  Point2f piece_centroid;
  Scalar colour(int(0.741 * 255), int(0.447 * 255), int(0.000 * 255));
  Scalar colour_2(0, 0, 255);
  sensor_msgs::ImagePtr vis_msg;
  vector< vector<DMatch> > knn_matches;
  // Good SURF matches, binned according to the piece in the template.
  vector< vector<DMatch> > good_matches;
  vector<DMatch> good_matches_combined;
  vector<KeyPoint> kp_piece;
  // SURF feature points in the puzzle piece,
  // binned according to the template piece they are matched to.
  vector< vector<Point2f> > pt_piece;
  // SURF feature points from the template,
  // binned according the piece they lie in.
  vector< vector<Point2f> > pt_template(this->piece_contours_f.size());

  // Update output message header and image.
  img_tf_msg.header.stamp = ros::Time::now();

  // Populate keypoints of piece.
  for(i = 0; i < msg->surf_key_points.size(); ++i)
  {
      kp_piece.push_back(KeyPoint(
            msg->surf_key_points[i].x,
            msg->surf_key_points[i].y,
            1.0));
  }

  // Initialize good matches and points.
  for(i = 0; i < this->piece_contours_f.size(); ++i)
  {
    good_matches.push_back(vector<DMatch>());
    pt_piece.push_back(vector<Point2f>());
    pt_template.push_back(vector<Point2f>());
  }

  // Compare with SURF features extracted from template image.
  // https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
  ROS_INFO_STREAM("Matching SURF features from image with template...");
  this->matcher->knnMatch(desc_piece, this->desc_template, knn_matches, 2);
  // ROS_INFO("Running comparison for good matches...");

  for(i = 0; i < knn_matches.size(); ++i)
  {
    // ROS_INFO_STREAM("Checking knn_matches[" << i << "] of size " << knn_matches[i].size() << "...");

    if((knn_matches[i][0].distance
          < dist_ratio_threshold * knn_matches[i][1].distance)
        && (knn_matches[i][0].trainIdx < this->kp_piece_indices.size())
        && (this->kp_piece_indices[knn_matches[i][0].trainIdx] < pt_piece.size())
        && (this->kp_piece_indices[knn_matches[i][0].trainIdx] < pt_template.size())
        && (knn_matches[i][0].queryIdx < msg->surf_key_points.size()))
    {
      // Determine which piece in the template this match throws.
      // ROS_INFO_STREAM("knn_matches[i][0].trainIdx  = " << knn_matches[i][0].trainIdx << "/" << this->kp_piece_indices.size() - 1);
      j = this->kp_piece_indices[knn_matches[i][0].trainIdx];
      // ROS_INFO_STREAM("j = " << j << "/" << pt_piece.size() - 1 << " " << j << "/" << pt_template.size() - 1);
      // Populate the good matches, piece points,
      // and template points accordingly.
      good_matches[j].push_back(knn_matches[i][0]);
      good_matches_combined.push_back(knn_matches[i][0]);
      // ROS_INFO_STREAM("knn_matches[i][0].queryIdx = " << knn_matches[i][0].queryIdx << "/" << msg->surf_key_points.size());
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

  // ROS_INFO("knn_matches successfully checked.");

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

  // Update likely piece index with estimated index.
  stringstream frame_str;
  frame_str << likely_piece_index;
  img_tf_msg.header.frame_id = frame_str.str();

  // Draw the piece contour in the visualization template.
  drawContours(template_vis, this->piece_contours_i, likely_piece_index, colour, 2);

  if((pt_template[likely_piece_index].size() > 0)
      && (pt_piece[likely_piece_index].size() > 0))
  {
    // Compute the homographic transformation.
    ROS_INFO_STREAM(
        "Computing homographic transformation from piece ("
        << pt_piece[likely_piece_index].size()
        << " features) to template ("
        << pt_template[likely_piece_index].size()
        << " features)...");
    h_inv = findHomography(
        pt_template[likely_piece_index],
        pt_piece[likely_piece_index],
        CV_RANSAC);

    if(h_inv.rows > 0)
    {
      // Update h_inv in the output message.
      /*
      ROS_INFO("Populating homographic transformation elements...");
      cv_bridge::CvImage(
          img_tf_msg.header,
          sensor_msgs::image_encodings::TYPE_32FC1,
          h_inv).toImageMsg(
            img_tf_msg.homographic_transform); */
      //ROS_INFO("Instantiating list of transformed points...");
      vector<Point2f> transformed_points;
      /*
         ROS_INFO_STREAM("Running perspective transformation on "
         << this->piece_central_points[likely_piece_index].size()
         << " points with homography transformation of size "
         << h_inv.size() << "..."); */
      perspectiveTransform(
          this->piece_central_points[likely_piece_index],
          transformed_points,
          h_inv);

      /*
         ROS_INFO_STREAM(
         "h_inv: " << h_inv.at<double>(0, 0) << ", " << h_inv.at<double>(0, 1)
         << ", " << h_inv.at<double>(0, 2) << ", " << h_inv.at<double>(0, 3) << "\n"
         << "       " << h_inv.at<double>(1, 0) << ", " << h_inv.at<double>(1, 1)
         << ", " << h_inv.at<double>(1, 2) << ", " << h_inv.at<double>(1, 3) << "\n"
         << "       " << h_inv.at<double>(2, 0) << ", " << h_inv.at<double>(2, 1)
         << ", " << h_inv.at<double>(2, 2) << ", " << h_inv.at<double>(2, 3) << "\n"
         << "       " << h_inv.at<double>(2, 0) << ", " << h_inv.at<double>(2, 1)
         << ", " << h_inv.at<double>(2, 2) << ", " << h_inv.at<double>(2, 3) << "\n");
         */

      piece_centroid.x = msg->centroid_px.x;
      piece_centroid.y = msg->centroid_px.y;

      circle(img_piece, piece_centroid, 8.0, colour_2, 2, 8, 0);
      double pixel_threshold = 10.0;

      // Run a quality check on the transformation, comparing centroids.
      if((piece_centroid.x - transformed_points[0].x < pixel_threshold)
          && (piece_centroid.x - transformed_points[0].x > -pixel_threshold)
          && (piece_centroid.y - transformed_points[0].y < pixel_threshold)
          && (piece_centroid.y - transformed_points[0].y > -pixel_threshold))
      {
        img_tf_msg.pose.position.x = (double) transformed_points[0].x - (double) img_piece.cols;
        img_tf_msg.pose.position.y = (double) transformed_points[0].y - (double) img_piece.rows;
        img_tf_msg.pose.position.z = sqrt(
            (img_tf_msg.pose.position.x * img_tf_msg.pose.position.x)
            + (img_tf_msg.pose.position.y * img_tf_msg.pose.position.y));

        // Compute the x-axis.
        ROS_INFO_STREAM("Computing x-axis of piece " << likely_piece_index << "...");
        Point3f x_axis = Point3f(
            transformed_points[1].x - transformed_points[0].x,
            transformed_points[1].y - transformed_points[0].y,
            0.0);
        double x_axis_norm = sqrt((x_axis.x * x_axis.x) + (x_axis.y * x_axis.y));
        x_axis.x /= x_axis_norm;
        x_axis.y /= x_axis_norm;

        // Compute the z-axis
        Point3f z_axis = Point3f(0.0, 0.0, 1.0);

        // Compute the y-axis.
        ROS_INFO_STREAM("Computing y-axis of piece " << likely_piece_index << "...");
        Point3f y_axis = z_axis.cross(x_axis);
        img_tf_msg.pose.orientation = AxesToQuaternion(x_axis, y_axis, z_axis);

        // Update output message with transformed points.
        /*
        for(i = 0; i < transformed_points.size(); ++i)
        {
          pt_temp.x = transformed_points[i].x;
          pt_temp.y = transformed_points[i].y;
          img_tf_msg.transformed_points.push_back(pt_temp);
        } */

        // Update the visualization message with the x and y axes.
        circle(img_piece, transformed_points[0], 10, colour, 2, 8, 0);
        line(img_piece, transformed_points[0], transformed_points[1], colour, 2, 8, 0);
        line(img_piece, transformed_points[0], transformed_points[2], colour, 2, 8, 0);

        // Publish the output message.
        ROS_INFO("Publishing homography and transformed central points...");
        this->transform_pub.publish(img_tf_msg);
        this->robot_stationary = false;
      }
      else
      {
        ROS_WARN_STREAM(
            "Transformed centroid (" << transformed_points[0].x << ", " << transformed_points[0].y
            << ") outside of threshold from detected centroid ("
            << piece_centroid.x << ", " << piece_centroid.y << ").");
      }
    }
    else
    {
      ROS_WARN("Transformation `h_inv` could not be computed, possibly due to too few features matched.");
    }
  }
  else
  {
    ROS_WARN_STREAM("Cannot compute homographic transform without any features.");
  }

  // Publish the visualization message.
  //ROS_INFO("Drawing SURF matches on visualization image...");
  drawMatches(
      img_piece, kp_piece,
      template_vis, this->kp_template,
      good_matches_combined, img_vis,
      Scalar::all(-1), Scalar::all(-1), vector<char>(),
      DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  vis_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img_vis).toImageMsg();
  //ROS_INFO("Publishing image for visualization...");
  this->vis_pub.publish(vis_msg);
}

/*
 * http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
 */
geometry_msgs::Quaternion AxesToQuaternion(Point3f x_axis, Point3f y_axis, Point3f z_axis)
{
  double rot_trace = x_axis.x + y_axis.y + z_axis.z;
  double q_norm_factor;
  geometry_msgs::Quaternion q;

  if(rot_trace > 0.0)
  {
    q_norm_factor = sqrt(rot_trace + 1.0) * 2; // q_norm_factor=4*q.w
    q.w = 0.25 * q_norm_factor;
    q.x = (y_axis.z - z_axis.y) / q_norm_factor;
    q.y = (z_axis.x - x_axis.z) / q_norm_factor;
    q.z = (x_axis.y - y_axis.x) / q_norm_factor;
  }
  else if((x_axis.x > y_axis.y)&(x_axis.x > z_axis.z))
  {
    q_norm_factor = sqrt(1.0 + x_axis.x - y_axis.y - z_axis.z) * 2; // q_norm_factor=4*q.x
    q.w = (y_axis.z - z_axis.y) / q_norm_factor;
    q.x = 0.25 * q_norm_factor;
    q.y = (y_axis.x + x_axis.y) / q_norm_factor;
    q.z = (z_axis.x + x_axis.z) / q_norm_factor;
  }
  else if(y_axis.y > z_axis.z)
  {
    q_norm_factor = sqrt(1.0 + y_axis.y - x_axis.x - z_axis.z) * 2; // q_norm_factor=4*q.y
    q.w = (z_axis.x - x_axis.z) / q_norm_factor;
    q.x = (y_axis.x + x_axis.y) / q_norm_factor;
    q.y = 0.25 * q_norm_factor;
    q.z = (z_axis.y + y_axis.z) / q_norm_factor;
  }
  else
  {
    q_norm_factor = sqrt(1.0 + z_axis.z - x_axis.x - y_axis.y) * 2; // q_norm_factor=4*q.z
    q.w = (x_axis.y - y_axis.x) / q_norm_factor;
    q.x = (z_axis.x + x_axis.z) / q_norm_factor;
    q.y = (z_axis.y + y_axis.z) / q_norm_factor;
    q.z = 0.25 * q_norm_factor;
  }

  return q;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "feature_matcher_node");
  ros::NodeHandle nh;
  FeatureMatcher p(nh, 400.0);
  ros::spin();

  return 0;
}

