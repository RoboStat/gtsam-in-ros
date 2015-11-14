#ifndef LANDMARK_H
#define LANDMARK_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>


class Landmark
{
    public:
        Landmark();
        Landmark(cv::KeyPoint kp_l, cv::KeyPoint kp_r, cv::Mat dscp_l, cv::Mat dscp_r);
        virtual ~Landmark();
        static int totalNumofLandmarks;

        const int id = totalNumofLandmarks;
        bool is_triangulated;

        // Changing attributes
        cv::KeyPoint keypoint_l;
        cv::KeyPoint keypoint_r;
        cv::Mat descriptor_l;
        cv::Mat descriptor_r;

        std::vector <double> coord_3D;
        cv::Mat patch_cv;

        // History attributes
        std::vector <cv::KeyPoint> traceHistory_l;
        std::vector <cv::KeyPoint> traceHistory_r;
        std::vector <int> traceFrameNum;

        void UpdateKeypoint(cv::KeyPoint kp_l, cv::KeyPoint kp_r, int nFrame);
        void UpdateKeypoint(cv::Point2f kp_l, cv::Point2f kp_r, int nFrame);

    protected:
    private:
};

#endif // LANDMARK_H
