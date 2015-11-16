/*
 * camera.h
 *
 *  Created on: Nov 8, 2015
 *      Author: wenda
 */

#ifndef INCLUDE_SLAM_MAIN_CAMERA_H_
#define INCLUDE_SLAM_MAIN_CAMERA_H_

#include <opencv2/core.hpp>

struct Camera {

	Camera(){
		focalLength = 164.25;
		baseline = 0.1621;
		principalPoint = cv::Point2f(213.51,118.43);
		intrinsic=(cv::Mat_<float>(3,3)<<
				focalLength,0,principalPoint.x,
				0,focalLength,principalPoint.y,
				0,0,1);
	}

	float focalLength;
	cv::Point2f principalPoint;
	float baseline;
	cv::Mat intrinsic;
};



#endif /* INCLUDE_SLAM_MAIN_CAMERA_H_ */
