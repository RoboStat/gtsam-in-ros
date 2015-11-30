#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>

/** ros include */
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

/** local include */
#include "VisualOdometry.h"
#include "GlobalVariables.h"
#include "Visualization.h"


using namespace cv;
using namespace std;
int Landmark::totalNumofLandmarks = 0;


int main(int argc,char* argv[])
{
//    testRun();
//    testRunGridDetect();
    ros::init(argc,argv,"gtsam_viz");
    ros::NodeHandle n;
    ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker",10);


    visualization_msgs::Marker pose;
    visualization_msgs::Marker landmark_location;

    vector<geometry_msgs::Point> path_;
    vector<geometry_msgs::Point> landmark_;
    drawColor(pose,path_,"poses",RED);
    drawColor(landmark_location,landmark_,"landmark",GREEN);
    //while(!ros::ok());
    marker_pub.publish(pose);
    marker_pub.publish(landmark_location);
    ros::Duration(1).sleep();

    VisualOdometry visualOdometry;
    visualOdometry.Initialize();

    for (int i=0;i<TOTAL_NUM_FRAMES;i++){
        if (i > 1120) {
            waitKey(0);
        }
        visualOdometry.Start_SAM();
        gtsam::Values pose_values = visualOdometry.sam().getInitialEstimate().filter<gtsam::Pose3>();
        path_.clear();
        for (size_t p=0; p<=i; ++p)
        {
            geometry_msgs::Point pt_;
            pt_.x = pose_values.at<gtsam::Pose3>(gtsam::Symbol('x',p)).translation().x();
            pt_.y = pose_values.at<gtsam::Pose3>(gtsam::Symbol('x',p)).translation().y();
            pt_.z = pose_values.at<gtsam::Pose3>(gtsam::Symbol('x',p)).translation().z();
            path_.push_back(pt_);
        }

        gtsam::Values landmark_values = visualOdometry.sam().getInitialEstimate().filter<gtsam::Point3>();

        landmark_.clear();
        for (auto p=landmark_values.begin(); p!=landmark_values.end(); ++p)
        {
            geometry_msgs::Point pt_;
            gtsam::Point3 loc_= static_cast<gtsam::Point3&>(p->value);
            pt_.x = loc_.x();
            pt_.y = loc_.y();
            pt_.z = loc_.z();
            landmark_.push_back(pt_);
        }


        pose.action=landmark_location.action=2;
        marker_pub.publish(pose);
        marker_pub.publish(landmark_location);
        drawColor(pose,path_,"poses",RED);
        drawColor(landmark_location,landmark_,"landmark",GREEN);
        marker_pub.publish(pose);
        marker_pub.publish(landmark_location);

    }
















    gtsam::Values pose_values = visualOdometry.sam().getInitialEstimate().filter<gtsam::Pose3>();
    pose_values.print("Final camera poses:\n");
    gtsam::Values landmark_values = visualOdometry.sam().getInitialEstimate().filter<gtsam::Point3>();
    pose_values.print("Final landmark poses:\n");
    visualOdometry.PrintTrajectory("out.txt");

    std::ofstream myfile;
    myfile.open ("out1.txt");

    for (unsigned int i=0; i<TOTAL_NUM_FRAMES; i++) {
        myfile << pose_values.at<gtsam::Pose3>(gtsam::Symbol('x',i)).translation().x() << " "
               << pose_values.at<gtsam::Pose3>(gtsam::Symbol('x',i)).translation().y() << " "
               << pose_values.at<gtsam::Pose3>(gtsam::Symbol('x',i)).translation().z() << " " << std::endl;
    }

    myfile.close();


    waitKey(0);// wait for a keystroke in the window
    return 0;
}
