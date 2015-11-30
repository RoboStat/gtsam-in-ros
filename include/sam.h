/** This class is for running smoothing and mapping with the VO data
 *  Functions include, factor graph management and optimization calls
 *
 *  Author: Yuhan Long <yuhanlon@andrew.cmu,edu>
 *  11/20/2015
 */


#ifndef SAM_H_
#define SAM_H_

/** GTSam include */
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/PriorFactor.h>
/** c++ include */
#include <string>
#include <fstream>
#include <iostream>


/** VO include */
#include "Landmark.h"
#include "camera.h"
#include "VisualOdometry.h"


using namespace std;
using namespace gtsam;
class Sam
{
friend class VisualOdometry;
public:
    Sam():
        parameters(ISAM2GaussNewtonParams(),0.01,1,true,false,ISAM2Params::CHOLESKY,true, DefaultKeyFormatter),
        isam(parameters),
        K(new Cal3_S2Stereo(164.25,164.25,0,213.5,118.43,0.162))
    {
    }

    static gtsam::Rot3 cv2gtsamR(const cv::Mat &R)
    {
        gtsam::Rot3 rot_(R.at<float>(0,0), R.at<float>(0,1),R.at<float>(0,2),
                        R.at<float>(1,0),R.at<float>(1,1),R.at<float>(1,2),
                        R.at<float>(2,0),R.at<float>(2,1),R.at<float>(2,2));
        return rot_;
    }
    static void cv2gtsamT(const cv::Mat &t, gtsam::Point3 &trans)
    {
        gtsam::Point3 trans_(t.at<float>(0,0),t.at<float>(1,0),t.at<float>(2,0));
        trans=trans_;
    }

    static void gtsam2cvR(const gtsam::Rot3 &R, cv::Mat &rot)
    {
        /** [TODO] make safe conversion */
        gtsam::Matrix3 mat_ = R.matrix();
        float init_[9] = {mat_(0,0),mat_(0,1),mat_(0,2),mat_(1,0),mat_(1,1),mat_(1,2),mat_(2,0),mat_(2,1),mat_(2,2)};
        cv::Mat rot_(3,3,CV_32FC1,init_);
        rot_.copyTo(rot);
    }

    static void gtsam2cvT(const gtsam::Point3 &t, cv::Mat &trans)
    {
        float init_[3] = {t.x(),t.y(),t.z()};
        cv::Mat trans_(3,1,CV_32FC1,init_);
        trans_.copyTo(trans);
    }
    const Values getInitialEstimate() const
    {
        return initial_estimate;
    }
private:
    Values initial_estimate;
    NonlinearFactorGraph graph;
    Values initial_estimate_history;

    ISAM2Params parameters;
    ISAM2 isam;
    const Cal3_S2Stereo::shared_ptr K;//=Cal3_S2Stereo(164.25,164.25,0,213.5,118.43,0.162);
    const noiseModel::Isotropic::shared_ptr model = noiseModel::Isotropic::Sigma(3,10);
};

#endif /** SAM_H_ */
