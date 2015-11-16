#include <opencv2/core/core.hpp>

#define MIN_NUM_LANDMARKS 40
#define TRACKER_ERR_THRESHOLD 10000
#define STARTING_FRAME 0
#define TOTAL_NUM_FRAMES 4000

#define LEFT_PATH "cmu_16662_p3/NSHLevel2_Images/left"
#define RIGHT_PATH "cmu_16662_p3/NSHLevel2_Images/right"
#define GRID_SIZE 3

#define DEBUG 1
#define VISUALIZATION 0


//double stereoMatrix[3][4] = {1, 0, 0, 20, 0, 1, 0, 0, 0, 0, 1, 0};
//const cv::Mat CAMERA_homography (3, 4, CV_64F, stereoMatrix);

const float inlier_threshold = 20.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
