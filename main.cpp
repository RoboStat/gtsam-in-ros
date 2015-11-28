#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>

#include "VisualOdometry.h"
#include "GlobalVariables.h"

using namespace cv;
int Landmark::totalNumofLandmarks = 0;

void testRun();
void testRunGridDetect();

int main()
{
//    testRun();
//    testRunGridDetect();

    VisualOdometry visualOdometry;
    visualOdometry.Initialize();

    for (int i=0;i<TOTAL_NUM_FRAMES;i++){
        if (i == 640) {
            waitKey(0);
        }
        visualOdometry.Start_SAM();
    }


    gtsam::Values pose_values = visualOdometry.sam().getInitialEstimate().filter<gtsam::Pose3>();
    pose_values.print("Final camera poses:\n");
    visualOdometry.PrintTrajectory("out.txt");

    waitKey(0);// wait for a keystroke in the window
    return 0;
}


void testRun()
{
    Mat image_l, image_r;// new blank image
    image_l = cv::imread("cmu_16662_p3/NSHLevel2_Images/left0001.jpg", 0);// read the file
    image_r = cv::imread("cmu_16662_p3/NSHLevel2_Images/right0001.jpg", 0);// read the file

    if (image_l.empty())
    {
        std::cout << "Image loading err" << std::endl;
        exit(0);
    }

    Mat mask1;
    mask1.create(image_l.size(), CV_8UC1);
    mask1.zeros(image_l.size(), CV_8UC1);
    Mat output1;
    output1.create(image_l.size(), CV_8UC1);

    Mat mask2;
    mask2.create(image_r.size(), CV_8UC1);
    mask2.zeros(image_r.size(), CV_8UC1);
    Mat output2;
    output2.create(image_r.size(), CV_8UC1);

    ////// Finding features
    std::vector <KeyPoint> keypoints_l, keypoints_r;
    Mat dscp1, dscp2;

    ////// Matching features
    Mat homography;
    FileStorage fs("cmu_16662_p3/H1to3p.xml", FileStorage::READ);
    fs.getFirstTopLevelNode() >> homography;

//    double stereoMatrix[3][3] = {1, 0, 1, 0, 1, 0, 0, 0, 1};
//    homography = Mat(3, 3, CV_32FC2, stereoMatrix);

//    if (homography.empty())
//    {
//        std::cout << "Matrix loading err" << std::endl;
//        exit(0);
//    }

    const float inlier_threshold = 100.5f; // Distance threshold to identify inliers
    const float nn_match_ratio = 1.0f;   // Nearest neighbor matching ratio

    Ptr<ORB> detector = ORB::create();
    detector->setPatchSize(5);
    detector->setEdgeThreshold(0);

    cv::Range rangeCol = cv::Range(0, 50);
    cv::Range rangeRow = cv::Range(0, 90);
    cv::Mat subI_l (image_l, rangeRow, rangeCol);
    cv::Mat subI_r (image_r, rangeRow, rangeCol);




    detector->detectAndCompute(subI_l, noArray(), keypoints_l, dscp1);
    detector->detectAndCompute(subI_l, noArray(), keypoints_r, dscp2);

    BFMatcher matcher(NORM_HAMMING);
    std::vector< std::vector<DMatch> > nn_matches;
    matcher.knnMatch(dscp1, dscp2, nn_matches, 2);

    std::vector<KeyPoint> matched1, matched2, inliers1, inliers2;
    std::vector<DMatch> good_matches;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(keypoints_l[first.queryIdx]);
            matched2.push_back(keypoints_r[first.trainIdx]);
        }
    }

    for(unsigned i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_32FC2);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;

//        col = homography * col;
//        col /= col.at<double>(2);
        double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                            pow(col.at<double>(1) - matched2[i].pt.y, 2));

        std::cout << dist << std::endl;

        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }


//    drawKeypoints(image, keypoints, image);
    Mat res;
    drawMatches(subI_l, inliers1, subI_l, inliers2, good_matches, res);
    namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// create a window for display.
    imshow( "Display window", res );// show our image inside it.

//    Mat f;
//    std::vector <Point2f> leftPs, rightPs;
//    for (int i=0;i<inliers1.size();i++){
//        leftPs.push_back(inliers1[i].pt);
//        rightPs.push_back(inliers2[i].pt);
//    }
//
//    f = findFundamentalMat (leftPs, rightPs);
//    std::cout << f << std::endl;
}



void testRunGridDetect()
{
    VisualOdometry visualOdometry;
    visualOdometry.Initialize();

    Mat image_l, image_r;// new blank image
    image_l = cv::imread("cmu_16662_p3/NSHLevel2_Images/left0001.jpg", 0);// read the file
    std::vector <cv::KeyPoint> keypoints;
    cv::Mat dscp;
    visualOdometry.GridDetect(image_l, keypoints, dscp);
    visualOdometry.VisualizeLandmarks(image_l, image_l, keypoints, keypoints);
}
