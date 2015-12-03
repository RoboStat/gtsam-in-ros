#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

//#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
//#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <algorithm>

#include "Landmark.h"
#include "camera.h"
#include "sam.h"


using namespace gtsam;
class VisualOdometry
{
    public:
        VisualOdometry();
        virtual ~VisualOdometry();

        void Initialize();
        void FindNewLandmarks(std::vector <int>& newLandmarksInd,size_t number = 100);
        void ReadNextFrame();
        void TrackFeatures(std::vector <Landmark>& lms, cv::Mat i_l, cv::Mat i_l_last,
                           cv::Mat i_r, cv::Mat i_r_last, std::vector <int>& remain_landmarks);
        //void Start();
        void Start_SAM();

        void VisualizeLandmarks(cv::Mat image_l, cv::Mat image_r, std::vector <Landmark> lm);
        void VisualizeLandmarks(cv::Mat image_l, cv::Mat image_r,
                                std::vector <cv::KeyPoint> inliers1, std::vector <cv::KeyPoint> inliers2);

        void GridDetect(cv::Mat image, std::vector <cv::KeyPoint>& keyPointSet, cv::Mat& dscp);
        void GridDetectAndMatch(cv::Mat image_l, cv::Mat image_r,
                                std::vector <cv::KeyPoint>& keyPointSet_l, std::vector <cv::KeyPoint>& keyPointSet_r,
                                cv::Mat& dscp_l, cv::Mat& dscp_r, std::vector <Landmark>& landmarkCandidateSet,size_t number);
        void InitializeMotionEstimation();
        //void MotionEstimation(std::vector <int> landmarkInd);
        void MotionEstimation(std::vector <int> landmarkInd, std::vector<int> newLandmarkInd);
        void PrintTrajectory(char* filename);

        const Sam sam() const
        {
            return sam_;
        }

        /** GTSAM Utilities */
        /*
        inline
        void updateGTSAMPoseValue(gtsam::Values &init_estimate, const size_t &x, const cv::Mat &R, const cv::Mat d&t)
        {
            gtsam::Pose3 p= init_estimate.at<gtsam::Pose3>(gtsam::Symbol('x',x-1));
            initial_estimate.insert(Symbol('x', x), p.transform_to(Pose3(R,t)));
        }

        inline
        void updateGTSAMLandmarkValue(gtsam::Values &init_estimate, const size_t &x, const size_t &l, const gtsam::Point3 &point);
        {

            initial_estimate.insert(Symbol('l', l), point);
        }

        /**
        void updateGTSAMGraphFactor(gtsam::NonlinearFactorGraph &graph);
        {

        }
        */
    protected:
        void FindFeatures(cv::Mat image,
                          std::vector <cv::KeyPoint>& keypoints,
                          cv::Mat& dscp);
        void FindFeatures(cv::Mat image,
                          std::vector <cv::KeyPoint>& keypoints,
                          cv::Mat& dscp, size_t number);
        void MatchStereoFeatures(std::vector <cv::KeyPoint> keyPointSet_left, cv::Mat dscp1,
                                 std::vector <cv::KeyPoint> keyPointSet_right, cv::Mat dscp2,
                                 std::vector <Landmark>& landmarkCandidateSet);

    private:
        int grid_size;
        int num_of_frame;
        int num_of_landmarks_in_last_frame;
        std::vector <Landmark> landmarks;
        std::vector <std::vector <int> > landmarkMap; // Store the IDs of landmarks show up in each frame
        std::vector <int> key_frames;
        cv::Mat camera_essential_matrix;
        std::vector <cv::Mat> camera_R; // Left camera matrix
        std::vector <cv::Mat> camera_t;
        std::vector <cv::Mat> camera_R_incremental; // Left camera matrix
        std::vector <cv::Mat> camera_t_incremental;
        cv::Mat image_l;
        cv::Mat image_r;
        cv::Mat image_l_last;
        cv::Mat image_r_last;

        Camera camera_model;
        Sam sam_;
};

#endif // VISUALODOMETRY_H
