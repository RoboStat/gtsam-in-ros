#include <opencv2/core/core.hpp>

#define MIN_NUM_LANDMARKS 500
#define TRACKER_ERR_THRESHOLD 1000000
#define STARTING_FRAME 380
#define TOTAL_NUM_FRAMES 400

#define LEFT_PATH "/home/yuhan/cmu_16662_p3/NSHLevel2_Images/left"
#define RIGHT_PATH "/home/yuhan/cmu_16662_p3/NSHLevel2_Images/right"
#define GRID_SIZE 3

#define DEBUG 1
#define VISUALIZATION 1
#define WITH_GTSAM 1


//double stereoMatrix[3][4] = {1, 0, 0, 20, 0, 1, 0, 0, 0, 0, 1, 0};
//const cv::Mat CAMERA_homography (3, 4, CV_64F, stereoMatrix);

const float inlier_threshold = 20.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
