#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "Landmark.h"


class VisualOdometry
{
    public:
        VisualOdometry();
        virtual ~VisualOdometry();

        void Initialize();
        void FindNewLandmarks(std::vector <int>& newLandmarksInd);
        void ReadNextFrame();
        void TrackFeatures(std::vector <Landmark>& lms, cv::Mat i_l, cv::Mat i_l_last,
                           cv::Mat i_r, cv::Mat i_r_last, std::vector <int>& remain_landmarks);
        void Start();

        void VisualizeLandmarks(cv::Mat image_l, cv::Mat image_r, std::vector <Landmark> lm);
        void VisualizeLandmarks(cv::Mat image_l, cv::Mat image_r,
                                std::vector <cv::KeyPoint> inliers1, std::vector <cv::KeyPoint> inliers2);

        void GridDetect(cv::Mat image, std::vector <cv::KeyPoint>& keyPointSet, cv::Mat& dscp);
        void GridDetectAndMatch(cv::Mat image_l, cv::Mat image_r,
                                std::vector <cv::KeyPoint>& keyPointSet_l, std::vector <cv::KeyPoint>& keyPointSet_r,
                                cv::Mat& dscp_l, cv::Mat& dscp_r, std::vector <Landmark>& landmarkCandidateSet);

    protected:
        void FindFeatures(cv::Mat image,
                          std::vector <cv::KeyPoint>& keypoints,
                          cv::Mat& dscp);

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
        cv::Mat image_l;
        cv::Mat image_r;
        cv::Mat image_l_last;
        cv::Mat image_r_last;
};

#endif // VISUALODOMETRY_H
