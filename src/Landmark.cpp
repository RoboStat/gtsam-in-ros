#include "Landmark.h"
#include "GlobalVariables.h"

Landmark::Landmark()
{
    //ctor
    totalNumofLandmarks += 1;
}

Landmark::Landmark(cv::KeyPoint kp_l, cv::KeyPoint kp_r, cv::Mat dscp_l, cv::Mat dscp_r)
{
    this->totalNumofLandmarks += 1;
    this->keypoint_l = kp_l;
    this->keypoint_r = kp_r;
    this->descriptor_l = dscp_l;
    this->descriptor_r = dscp_r;

    this->is_triangulated = false;
}

Landmark::~Landmark()
{
    //dtor
}

void Landmark::UpdateKeypoint(cv::KeyPoint kp_l, cv::KeyPoint kp_r, int nFrame)
{
    this->keypoint_l = kp_l;
    this->keypoint_r = kp_r;
    this->traceHistory_l.push_back(kp_l);
    this->traceHistory_r.push_back(kp_r);
    this->traceFrameNum.push_back(nFrame);
}

void Landmark::UpdateKeypoint(cv::Point2f kp_l, cv::Point2f kp_r, int nFrame)
{
    cv::KeyPoint kp_ll, kp_rr;
    kp_ll.pt = kp_l;
    kp_rr.pt = kp_r;
    this->keypoint_l = kp_ll;
    this->keypoint_r = kp_rr;
    this->traceHistory_l.push_back(kp_ll);
    this->traceHistory_r.push_back(kp_rr);
    this->traceFrameNum.push_back(nFrame);
}
