#include "VisualOdometry.h"
#include "GlobalVariables.h"

#include <iomanip>





VisualOdometry::VisualOdometry()
{
    //_sam();
    //ctor
}

VisualOdometry::~VisualOdometry()
{
    //dtor
}

void VisualOdometry::Initialize()
{
    num_of_frame = 0;
    num_of_landmarks_in_last_frame = 0;
    grid_size = GRID_SIZE;
}

void VisualOdometry::FindNewLandmarks(std::vector <int>& newLandmarksInd)
{
    // Finding keypoints in grids
    std::vector <cv::KeyPoint> keyPointSet_left, keyPointSet_right;
    cv::Mat dscp_left, dscp_right;
    std::vector <Landmark> landmarkCandidateSet;

//    GridDetect(image_l, keyPointSet_left, dscp_left);
//    GridDetect(image_r, keyPointSet_right, dscp_right);
//
//    // Match stereo features and find landmark candidates
//    MatchStereoFeatures(keyPointSet_left, dscp_left, keyPointSet_right, dscp_right, landmarkCandidateSet);

    GridDetectAndMatch(image_l, image_r, keyPointSet_left, keyPointSet_right, dscp_left, dscp_right, landmarkCandidateSet);

#if DEBUG
    std::cout << "Total " << keyPointSet_left.size() << " keypoints in left image" << std::endl;
    std::cout << "Total " << keyPointSet_right.size() << " keypoints in right image" << std::endl;
#endif // DEBUG

#if DEBUG
    std::cout << landmarkCandidateSet.size() << " candidate landmarks" << std::cout;
    VisualizeLandmarks(image_l, image_r, landmarkCandidateSet);
#endif // DEBUG

    for (unsigned int i=0; i<landmarkCandidateSet.size(); i++){
        this->landmarks.push_back(landmarkCandidateSet[i]);
        newLandmarksInd.push_back(landmarks.size()-1);
    }
}

void VisualOdometry::GridDetect(cv::Mat image, std::vector <cv::KeyPoint>& keyPointSet, cv::Mat& dscp)
{
    cv::Size sizeI = image.size();
    cv::Range rangeRow, rangeCol;
    for (int i=0;i<grid_size;i++){
        for (int j=0;j<grid_size;j++){
            rangeCol = cv::Range(j*(sizeI.width/grid_size), std::min((j+1)*(sizeI.width/grid_size)-1,sizeI.width));
            rangeRow = cv::Range(i*(sizeI.height/grid_size),std::min((i+1)*(sizeI.height/grid_size)-1,sizeI.height));
            cv::Mat subI (image, rangeRow, rangeCol);

#if DEBUG
#endif // DEBUG

            std::vector <cv::KeyPoint> subKeyPoints;
            cv::Mat subDscp;
            FindFeatures(subI, subKeyPoints, subDscp);
            std::cout << subKeyPoints.size() << std::endl;
            for (auto it = subKeyPoints.begin(); it != subKeyPoints.end(); it++){
                it->pt.x += j*sizeI.width/grid_size;
                it->pt.y += i*sizeI.height/grid_size;
                keyPointSet.push_back(*it);
            }
            dscp.push_back(subDscp);
        }
    }
}

void VisualOdometry::GridDetectAndMatch(cv::Mat image_l, cv::Mat image_r,
                                        std::vector <cv::KeyPoint>& keyPointSet_l, std::vector <cv::KeyPoint>& keyPointSet_r,
                                        cv::Mat& dscp_l, cv::Mat& dscp_r, std::vector <Landmark>& landmarkCandidateSet)
{
    cv::Size sizeI = image_l.size();
    cv::Range rangeRow, rangeCol;
    for (int i=0;i<grid_size;i++){
        for (int j=0;j<grid_size;j++){
            rangeCol = cv::Range(j*(sizeI.width/grid_size), std::min((j+1)*(sizeI.width/grid_size)-1, sizeI.width));
            rangeRow = cv::Range(i*(sizeI.height/grid_size), std::min((i+1)*(sizeI.height/grid_size)-1, sizeI.height));
            cv::Mat subI_l (image_l, rangeRow, rangeCol);
            cv::Mat subI_r (image_r, rangeRow, rangeCol);

#if DEBUG
#endif // DEBUG

            std::vector <cv::KeyPoint> subKeyPoints_l, subKeyPoints_r;
            cv::Mat subDscp_l, subDscp_r;
            FindFeatures(subI_l, subKeyPoints_l, subDscp_l);
            FindFeatures(subI_r, subKeyPoints_r, subDscp_r);
            //std::cout << subKeyPoints.size() << std::endl;
            for (auto it = subKeyPoints_l.begin(); it != subKeyPoints_l.end(); it++){
                it->pt.x += j*sizeI.width/grid_size;
                it->pt.y += i*sizeI.height/grid_size;
                keyPointSet_l.push_back(*it);
            }
            dscp_l.push_back(subDscp_l);

            for (auto it = subKeyPoints_r.begin(); it != subKeyPoints_r.end(); it++){
                it->pt.x += j*sizeI.width/grid_size;
                it->pt.y += i*sizeI.height/grid_size;
                keyPointSet_r.push_back(*it);
            }
            dscp_r.push_back(subDscp_r);

            std::vector <Landmark> subLandmarkCandidateSet;
            MatchStereoFeatures(subKeyPoints_l, subDscp_l, subKeyPoints_r, subDscp_r, subLandmarkCandidateSet);
            //VisualizeLandmarks(image_l, image_r, subLandmarkCandidateSet);

            for (unsigned int i=0; i<subLandmarkCandidateSet.size(); i++){
                landmarkCandidateSet.push_back(subLandmarkCandidateSet[i]);
            }
        }
    }
#if DEBUG
            std::cout << "Exiting GridDetectAndMatch" << std::endl;
#endif // DEBUG
}

void VisualOdometry::MatchStereoFeatures(std::vector <cv::KeyPoint> keyPointSet_left,
                                         cv::Mat dscp1,
                                         std::vector <cv::KeyPoint> keyPointSet_right,
                                         cv::Mat dscp2,
                                         std::vector <Landmark>& landmarkCandidateSet)
{
    using namespace cv;

    if (keyPointSet_left.size() == 0 || keyPointSet_right.size() == 0)
        return;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector< std::vector<cv::DMatch> > nn_matches;
    matcher.knnMatch(dscp1, dscp2, nn_matches, 2);

#if DEBUG
    std::cout << "Found feature match: " << nn_matches.size() << std::endl;
#endif // DEBUG

    std::vector<cv::KeyPoint> matched1, matched2, inliers1, inliers2;
    std::vector <cv::Mat> matched_dscp1, matched_dscp2, inlier_dscp1, inlier_dscp2;
    std::vector<cv::DMatch> good_matches;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(keyPointSet_left[first.queryIdx]);
            matched2.push_back(keyPointSet_right[first.trainIdx]);
            matched_dscp1.push_back(dscp1.row(first.queryIdx));
            matched_dscp2.push_back(dscp2.row(first.trainIdx));
        }
    }
    #if DEBUG
            std::cout << "Exiting GridDetectAndMatch" << std::endl;
#endif // DEBUG

    for(unsigned i = 0; i < matched1.size(); i++) {
        double dist = sqrt( pow(matched1[i].pt.x - matched2[i].pt.x, 2) +
                            pow(matched1[i].pt.y - matched2[i].pt.y, 2));

//        std::cout << dist << std::endl;

        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            inlier_dscp1.push_back(matched_dscp1[i]);
            inlier_dscp2.push_back(matched_dscp2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    #if DEBUG
            std::cout << "Exiting GridDetectAndMatch" << std::endl;
            std::cout << "dscp size: " << dscp1.size().height << std::endl;
            std::cout << "inlier size: " << inliers1.size() << std::endl;
#endif // DEBUG

    for(unsigned int i = 0; i < inliers1.size(); i++) {
        Landmark new_landmark (inliers1[i], inliers2[i], inlier_dscp1[i], inlier_dscp2[i]);
        landmarkCandidateSet.push_back(new_landmark);
    }
#if DEBUG
            std::cout << "Exiting MatchStereoFeature" << std::endl;
#endif // DEBUG
}

void VisualOdometry::FindFeatures(cv::Mat image,
                                 std::vector <cv::KeyPoint>& keypoints,
                                 cv::Mat& dscp)
{
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
//    detector->setPatchSize(5);
    detector->setEdgeThreshold(1);
    detector->detectAndCompute(image,cv::noArray(), keypoints, dscp);
//    for (int i=0;i<keypoints.size();i++){
//        Landmark new_lm (keypoints[i], dscp.row(i));
//        featurePointSet.push_back(new_lm);
//    }
//    return featurePointSet;

#if DEBUG
    //std::cout << "Found " << keypoints.size() << " keypoints in grid." << std::endl;
    //VisualizeLandmarks(image, image, keypoints, keypoints);
#endif // DEBUG
}

void VisualOdometry::ReadNextFrame()
{
    std::ostringstream ss_l, ss_r;
    ss_l << LEFT_PATH << std::setw(4) << std::setfill('0') << num_of_frame + STARTING_FRAME << ".jpg";
    ss_r << RIGHT_PATH << std::setw(4) << std::setfill('0') << num_of_frame + STARTING_FRAME << ".jpg";

    std::string filename_l = ss_l.str();
    std::string filename_r = ss_r.str();

    image_l_last = image_l;
    image_r_last = image_r;

#if DEBUG
    std::cout << "Reading " << ss_l.str() << std::endl;
#endif // DEBUG
    image_l = cv::imread(filename_l, 0);// read the file

#if DEBUG
    std::cout << "Reading " << ss_l.str() << std::endl;
#endif // DEBUG
    image_r = cv::imread(filename_r, 0);// read the file

    if (image_l.empty() || image_r.empty())
        std::cerr << "Error loading stereo image: " << num_of_frame << std::endl;
}

void VisualOdometry::TrackFeatures(std::vector <Landmark>& active_landmarks, cv::Mat i_l, cv::Mat i_l_last,
                                   cv::Mat i_r, cv::Mat i_r_last, std::vector <int>& remain_landmarks)
{
    using namespace cv;
    if (active_landmarks.size() == 0)
        return;

    std::vector <cv::Point2f> kp_l, kp_r, kp_l_last, kp_r_last;
    for (unsigned int i=0;i<active_landmarks.size();i++){
        kp_l_last.push_back(active_landmarks[i].keypoint_l.pt);
        kp_r_last.push_back(active_landmarks[i].keypoint_r.pt);
//        kp_l.push_back(active_landmarks[i].keypoint_l.pt);
//        kp_r.push_back(active_landmarks[i].keypoint_r.pt);
    }
    Mat status_l, err_l, status_r, err_r;

    calcOpticalFlowPyrLK(i_l_last, i_l, kp_l_last, kp_l, status_l, err_l, Size(21, 21));
    calcOpticalFlowPyrLK(i_r_last, i_r, kp_r_last, kp_r, status_r, err_r, Size(21, 21));

#if DEBUG
    std::cout << err_l.size() << std::endl;
    std::cout << err_r.size() << std::endl;
    std::cout << active_landmarks.size() << std::endl;
    std::cout << kp_l_last[0] << " " << kp_l[0] << std::endl;
#endif // DEBUG

    for (unsigned int i=0; i<active_landmarks.size(); i++){
        if (err_l.at<double>(i,1) < TRACKER_ERR_THRESHOLD && err_r.at<double>(i,1) < TRACKER_ERR_THRESHOLD){
            if (kp_l[i].x < 0 || kp_l[i].x > i_l.cols || kp_l[i].y < 0 || kp_l[i].y > i_l.rows ||
                kp_r[i].x < 0 || kp_r[i].x > i_r.cols || kp_r[i].y < 0 || kp_r[i].y > i_r.rows) {
                std::cout << "Throw away landmark " << kp_l[i] << "  " << kp_r[i] << std::endl;
            }
            else {
                landmarks[active_landmarks[i].id].UpdateKeypoint(kp_l[i], kp_r[i], num_of_frame);
                remain_landmarks.push_back(i);
            }
        }
    }
}

void VisualOdometry::InitializeMotionEstimation()
{
    // Finding the Essential matrix
    std::vector <cv::Point2f> points_l, points_r;
    for (unsigned int i=0;i<landmarks.size();i++){
        points_l.push_back(landmarks[i].keypoint_l.pt);
        points_r.push_back(landmarks[i].keypoint_r.pt);
    }

    camera_essential_matrix = cv::findEssentialMat(points_l, points_r, camera_model.focalLength, camera_model.principalPoint);

#if DEBUG
    std::cout << "E: " << camera_essential_matrix << std::endl;
    cv::Mat R, t;
    recoverPose(camera_essential_matrix, points_l, points_r, R, t);
    std::cout << "R: " << R << std::endl;
    std::cout << "t: " << t << std::endl;
#endif


    cv::Mat identity = cv::Mat::eye(3, 3, CV_32F);
    camera_R.push_back(identity);
    camera_t.push_back(cv::Mat::zeros(3, 1, CV_32F));

#if WITH_GTSAM
    using namespace gtsam;
    {
        //Eigen::Matrix3f rot_;
        //cv2eigen(camera_R,rot_);
        //camera_R.back();
        Rot3 rot_=Sam::cv2gtsamR(camera_R.back());
        Point3 trans_=Sam::cv2gtsamT(camera_t.back());
        cout<<"add initial pose"<<endl;
        sam_.initial_estimate.insert(Symbol('x', 0), Pose3(rot_,trans_));
        noiseModel::Diagonal::shared_ptr poseNoise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.3),Vector3::Constant(0.1))); // 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
        sam_.graph.push_back(PriorFactor<Pose3>(Symbol('x', 0), Pose3(rot_,trans_), poseNoise));

    }
#endif
}

void VisualOdometry::MotionEstimation(std::vector <int> landmarkInd)
/*
    landmarkInd stores the index of shared landmarks between last frame and this frame.
*/
{
    cv::Mat proj;
    std::vector <cv::Point2f> points_current_l, points_last_l, points_last_r;
    for (unsigned int i=0;i<landmarkInd.size();i++){
        int ind = landmarkInd[i];
        points_current_l.push_back(landmarks[ind].keypoint_l.pt);
        points_last_l.push_back(landmarks[ind].traceHistory_l.rbegin()[1].pt);
        points_last_r.push_back(landmarks[ind].traceHistory_r.rbegin()[1].pt);
    }

    cv::Mat camera_matrix_l, camera_matrix_r;
    cv::Mat rt_l, rt_r;
    cv::Mat d = (cv::Mat_<float>(3,1) << -camera_model.baseline, 0, 0);

    // Triangulate point in the last frame
    cv::Mat points_last_4D;

    // World frame triangulation method
    // cv::triangulatePoints(camera_matrix_l, camera_matrix_r, points_last_l, points_last_r, points_last_4D);

    // Incremental camera frame triangulation method
    cv::hconcat(camera_R[0], camera_t[0], rt_l);
    cv::hconcat(camera_R[0], camera_t[0] + d, rt_r);
    camera_matrix_l = camera_model.intrinsic*rt_l;
    camera_matrix_r = camera_model.intrinsic*rt_r;
    cv::triangulatePoints(camera_matrix_l, camera_matrix_r, points_last_l, points_last_r, points_last_4D);

    cv::Mat R_new_q, R_new_incremental, t_new_incremental, R_new, t_new;
    std::vector <cv::Point3f> points_last_3D;
    for (int i=0;i<points_last_4D.cols;i++) {
        cv::Point3f pt;
        pt.x =  points_last_4D.at<float>(0,i) / points_last_4D.at<float>(3,i);
        pt.y = points_last_4D.at<float>(1,i) / points_last_4D.at<float>(3,i);
        pt.z = points_last_4D.at<float>(2,i) / points_last_4D.at<float>(3,i);
        points_last_3D.push_back(pt);
    }

    std::vector <double> distCoeffs;
    cv::Mat pnpInliers;
    cv::solvePnPRansac (points_last_3D, points_current_l, camera_model.intrinsic, distCoeffs, R_new_q, t_new_incremental, false, 100, 1.0, 0.98, pnpInliers);
    cv::Rodrigues(R_new_q, R_new_incremental);
    R_new_incremental.convertTo(R_new_incremental, CV_32F);
    t_new_incremental.convertTo(t_new_incremental, CV_32F);
    camera_R_incremental.push_back(R_new_incremental);
    camera_t_incremental.push_back(t_new_incremental);

    R_new = R_new_incremental*camera_R.back() ;
    t_new = camera_t.back() + camera_R.back().inv()*t_new_incremental;
    camera_R.push_back(R_new);
    camera_t.push_back(t_new);

#if DEBUG
    cv::hconcat(R_new, t_new, proj);
    std::cout << "PnP Total num of feature points: " << points_current_l.size() << "  PnP num of Inliers: " << pnpInliers.size() << std::endl;
    std::cout << "New camera R: " << R_new << std::endl;
    std::cout << "New camera t: " << t_new << std::endl;
#endif

    std::cout << "New camera matrix: " << proj << std::endl;
}


void VisualOdometry::MotionEstimation(std::vector <int> landmarkInd,std::vector<int> newLandmarksInd)
{
    cv::Mat proj;
    std::vector <cv::Point2f> points_current_l, points_last_l, points_last_r;
    for (unsigned int i=0;i<landmarkInd.size();i++){
        int ind = landmarkInd[i];
        points_current_l.push_back(landmarks[ind].keypoint_l.pt);
        points_last_l.push_back(landmarks[ind].traceHistory_l.rbegin()[1].pt);
        points_last_r.push_back(landmarks[ind].traceHistory_r.rbegin()[1].pt);
    }

    cv::Mat camera_matrix_l, camera_matrix_r;
    cv::Mat rt_l, rt_r;
    cv::Mat d = (cv::Mat_<float>(3,1) << -camera_model.baseline, 0, 0);

    // Triangulate point in the last frame
    cv::Mat points_last_4D;

    // Incremental camera frame triangulation method
    cv::hconcat(camera_R[0], camera_t[0], rt_l);
    cv::hconcat(camera_R[0], camera_t[0] + d, rt_r);
    camera_matrix_l = camera_model.intrinsic*rt_l;
    camera_matrix_r = camera_model.intrinsic*rt_r;
    cv::triangulatePoints(camera_matrix_l, camera_matrix_r, points_last_l, points_last_r, points_last_4D);

    cv::Mat R_new_q, R_new_incremental, t_new_incremental, R_new, t_new;
    std::vector <cv::Point3f> points_last_3D;
    for (int i=0;i<points_last_4D.cols;i++) {
        cv::Point3f pt;
        pt.x =  points_last_4D.at<float>(0,i) / points_last_4D.at<float>(3,i);
        pt.y = points_last_4D.at<float>(1,i) / points_last_4D.at<float>(3,i);
        pt.z = points_last_4D.at<float>(2,i) / points_last_4D.at<float>(3,i);
        points_last_3D.push_back(pt);
    }

    std::vector <double> distCoeffs;
    cv::Mat pnpInliers;
    cv::solvePnPRansac (points_last_3D, points_current_l, camera_model.intrinsic, distCoeffs, R_new_q, t_new_incremental, false, 100, 1.0, 0.98, pnpInliers);
    cv::Rodrigues(R_new_q, R_new_incremental);
    R_new_incremental.convertTo(R_new_incremental, CV_32F);
    t_new_incremental.convertTo(t_new_incremental, CV_32F);
    camera_R_incremental.push_back(R_new_incremental);
    camera_t_incremental.push_back(t_new_incremental);

    /** This paret is for updating the gtsam graph */
    Pose3 last_pose;
    if (sam_.initial_estimate.exists(Symbol('x',camera_R.size()-1)))
        last_pose = sam_.initial_estimate.at<Pose3>(Symbol('x',camera_R.size()-1)); // pull the last pose out using the size the R matrix;
    else
        last_pose = sam_.initial_estimate_history.at<Pose3>(Symbol('x',camera_R.size()-1)); // pull the last pose out using the size the R matrix;

    Rot3 current_R = Sam::cv2gtsamR(R_new_incremental)*last_pose.rotation();
    Point3 current_t = last_pose.translation()+last_pose.rotation().inverse()*Sam::cv2gtsamT(t_new_incremental);
    sam_.initial_estimate.insert(Symbol('x',camera_R.size()),Pose3(current_R,current_t));

    cout<<"new landmark number: "<<newLandmarksInd.size()<<endl;



    R_new = R_new_incremental*camera_R.back() ;
    t_new = camera_t.back() + camera_R.back().inv()*t_new_incremental;
    camera_R.push_back(R_new);
    camera_t.push_back(t_new);

#if DEBUG
    cv::hconcat(R_new, t_new, proj);
    std::cout << "PnP Total num of feature points: " << points_current_l.size() << "  PnP num of Inliers: " << pnpInliers.size() << std::endl;
    std::cout << "New camera R: " << R_new << std::endl;
    std::cout << "New camera t: " << t_new << std::endl;
#endif

    std::cout << "New camera matrix: " << proj << std::endl;
}





void VisualOdometry::Start_SAM()
{
    num_of_frame += 1;
    ReadNextFrame();

    std::vector <Landmark> active_landmarks;
    if (landmarkMap.size()>0){
        std::cout << "Num of item in landmark map " << num_of_frame - 2 << " is " << landmarkMap[num_of_frame-2].size() << std::endl;
        for (unsigned int i=0;i<landmarkMap[num_of_frame-2].size();i++){
            active_landmarks.push_back(landmarks[landmarkMap[num_of_frame-2][i]]);
        }
    }

    std::vector <int> remain_landmarks;
    TrackFeatures( active_landmarks, image_l, image_l_last, image_r, image_r_last, remain_landmarks);



    std::vector <int> landmarkInd;
    for (unsigned int i=0; i<remain_landmarks.size(); i++){
        landmarkInd.push_back(active_landmarks[remain_landmarks[i]].id);
    }

    // Check if new landmarks needed
    std::vector <int> newLandmarksInd;
    if (remain_landmarks.size() < MIN_NUM_LANDMARKS){
        std::cout << "New Key Frame. " << std::endl;
        key_frames.push_back(num_of_frame);

        std::vector <int> landmarksInd_withNewLandmarks (landmarkInd.begin(), landmarkInd.end());
        FindNewLandmarks(newLandmarksInd);
        for(size_t i = 0; i<newLandmarksInd.size(); ++i)
        {
            landmarks[newLandmarksInd[i]].traceFrameNum.push_back(num_of_frame);
        }
        landmarksInd_withNewLandmarks.insert(landmarksInd_withNewLandmarks.end(),newLandmarksInd.begin(),newLandmarksInd.end());
        landmarkMap.push_back(landmarksInd_withNewLandmarks);
    }
    else {
        landmarkMap.push_back(landmarkInd);
    }

    // Triangulation and motion estimation
    if (num_of_frame == 1) InitializeMotionEstimation();
    else MotionEstimation(landmarkInd,newLandmarksInd);


    cout<<"new landmark number: "<<newLandmarksInd.size()<<endl;

    /** add new landmark into initial estimation and add the factors in the factor graph */

    Pose3 current_pose;

    if (sam_.initial_estimate.exists(Symbol('x',camera_R.size()-1)))
        current_pose = sam_.initial_estimate.at<Pose3>(Symbol('x',num_of_frame-1)); // pull the last pose out using the size the R matrix;
    else
        current_pose = sam_.initial_estimate_history.at<Pose3>(Symbol('x',num_of_frame-1)); // pull the last pose out using the size the R matrix;

/*
    if(!newLandmarksInd.empty())
    {
        cv::Mat current_R = Sam::gtsam2cvR(current_pose.rotation());
        cv::Mat current_t = Sam::gtsam2cvT(current_pose.translation());

        cv::Mat rt_l, rt_r;
        cv::hconcat(current_R, current_t, rt_l);

        cv::hconcat(current_R, current_t +cv::Mat(-camera_model.baseline, 0, 0), rt_r);

        std::vector <cv::Point2f> points_current_l, points_current_r;
        for (size_t i=0;i<newLandmarksInd.size();i++){
            int ind = newLandmarksInd[i];
            points_current_l.push_back(landmarks[ind].keypoint_l.pt);
            points_current_r.push_back(landmarks[ind].keypoint_r.pt);

            cv::Mat points_last_4D;
            cv::triangulatePoints(camera_model.intrinsic*rt_l, camera_model.intrinsic*rt_r, points_current_l, points_current_r, points_last_4D);

            double x_ =  points_last_4D.at<float>(0,0) / points_last_4D.at<float>(3,0);
            double y_ = points_last_4D.at<float>(1,0) / points_last_4D.at<float>(3,0);
            double z_ = points_last_4D.at<float>(2,0) / points_last_4D.at<float>(3,0);
            gtsam::Point3 pt(x_,y_,z_);
            //sam_.initial_estimate.insert(Symbol('l',ind),pt);
            if(ind==0)
            {
                noiseModel::Isotropic::shared_ptr pointNoise = noiseModel::Isotropic::Sigma(3, 0.1);
                //sam_.graph.push_back(gtsam::PriorFactor<Point3>(Symbol('l', 0), pt, pointNoise));
            }

            //sam_.graph.push_back(GenericStereoFactor<Pose3, Point3>(StereoPoint2(
            //    landmarks[ind].keypoint_l.pt.x, landmarks[ind].keypoint_r.pt.x,
            //    (landmarks[ind].keypoint_l.pt.y+landmarks[ind].keypoint_r.pt.y)/2),
            //    sam_.model,Symbol('x', num_of_frame-1),
            //    Symbol('l', ind), sam_.K));
        }
    }
*/
    /** add existing landmark in the factor graph */

    const size_t observe_threshold = 2;
    for (size_t i=0 ; i<landmarkInd.size() ; ++i)
    {

        size_t ind = landmarkInd[i];
        if (landmarks[ind].traceHistory_l.size()==observe_threshold)
        {
            /** triangulate the landmark */

            cv::Mat current_R(3,3,CV_32FC1);
            Sam::gtsam2cvR(current_pose.rotation(),current_R);
            cv::Mat current_t(3,1,CV_32FC1);
            Sam::gtsam2cvT(current_pose.translation(),current_t);

            cv::Mat rt_l, rt_r;
            cv::hconcat(current_R, current_t, rt_l);
            current_t.at<float>(0,0)=-camera_model.baseline+current_t.at<float>(0,0);
            cout<<current_t<<endl;
            cv::hconcat(current_R, current_t, rt_r);

            std::vector <cv::Point2f> points_current_l, points_current_r;

            points_current_l.push_back(landmarks[ind].keypoint_l.pt);
            points_current_r.push_back(landmarks[ind].keypoint_r.pt);

            cout<<rt_r<<endl;
            cv::Mat points_last_4D;
            /** repair the triangulation */
            cv::triangulatePoints(camera_model.intrinsic*rt_l, camera_model.intrinsic*rt_r, points_current_l, points_current_r, points_last_4D);
            double x_ =  points_last_4D.at<float>(0,0) / points_last_4D.at<float>(3,0);
            double y_ = points_last_4D.at<float>(1,0) / points_last_4D.at<float>(3,0);
            double z_ = points_last_4D.at<float>(2,0) / points_last_4D.at<float>(3,0);
            gtsam::Point3 pt(x_,y_,z_);
            cout<<x_<<" "<<y_<<" "<<z_<<endl;
            sam_.initial_estimate.insert(Symbol('l',ind),pt);
            cout<<"landmark_num: "<<sam_.initial_estimate.filter<Point3>().size()<<endl;
            if(sam_.initial_estimate.filter<Point3>().size()==1)
            {
                noiseModel::Isotropic::shared_ptr pointNoise = noiseModel::Isotropic::Sigma(3, 2);
                sam_.graph.push_back(gtsam::PriorFactor<Point3>(Symbol('l', 0), pt, pointNoise));
            }

            for(size_t history=0; history<landmarks[ind].traceFrameNum.size(); ++history)
            {
                cout<<landmarks[ind].traceHistory_l[history].pt.x<<endl;
                cout<<landmarks[ind].traceHistory_r[history].pt.x<<endl;
                sam_.graph.push_back(GenericStereoFactor<Pose3, Point3>(StereoPoint2(
                    landmarks[ind].traceHistory_l[history].pt.x, landmarks[ind].traceHistory_r[history].pt.x,
                    (landmarks[ind].traceHistory_l[history].pt.y+landmarks[ind].traceHistory_r[history].pt.y)/2),
                    sam_.model,Symbol('x', landmarks[ind].traceFrameNum[history]-1),
                    Symbol('l', ind), sam_.K));
                GenericStereoFactor<Pose3, Point3> f_(StereoPoint2(
                    landmarks[ind].traceHistory_l[history].pt.x, landmarks[ind].traceHistory_r[history].pt.x,
                    (landmarks[ind].traceHistory_l[history].pt.y+landmarks[ind].traceHistory_r[history].pt.y)/2),
                    sam_.model,Symbol('x', landmarks[ind].traceFrameNum[history]-1),
                    Symbol('l', ind), sam_.K);

                Vector e_=f_.evaluateError(current_pose,Point3(-1.01004255, -1.04808509, 1.52936518));
                cout<<"error estimate:"<<endl;
                cout<<e_<<endl;
            }
            break;

        }

        if (landmarks[ind].traceHistory_l.size()>observe_threshold)
        {
            sam_.graph.push_back(GenericStereoFactor<Pose3, Point3>(StereoPoint2(
                landmarks[ind].keypoint_l.pt.x, landmarks[ind].keypoint_r.pt.x,
                (landmarks[ind].keypoint_l.pt.y+landmarks[ind].keypoint_r.pt.y)/2),
                sam_.model,Symbol('x', num_of_frame-1),
                Symbol('l', ind), sam_.K));

        }

        //sam_.graph.push_back(GenericStereoFactor<Pose3, Point3>(StereoPoint2(
        //    landmarks[ind].keypoint_l.pt.x, landmarks[ind].keypoint_r.pt.x,
        //    (landmarks[ind].keypoint_l.pt.y+landmarks[ind].keypoint_r.pt.y)/2),
        //    sam_.model,Symbol('x', num_of_frame-1),
        //    Symbol('l', ind), sam_.K));
    }

    cerr<<"optimizing..."<<endl;

    //if ((key_frames.back()==num_of_frame)&&(num_of_frame!=1))
    if (num_of_frame==2)
    {
        cout<<"Total number of initial estimate:"<<sam_.initial_estimate.filter<Point3>().size()<<endl;
        //LevenbergMarquardtOptimizer optimizer = LevenbergMarquardtOptimizer(sam_.graph, sam_.initial_estimate);
        //Values new_estimate = optimizer.optimize();
        sam_.graph.print();
        Values res=sam_.initial_estimate.filter<Pose3>();
        res.print();
        sam_.isam.update(sam_.graph, sam_.initial_estimate);
        //sam_.isam.update();
        //Values new_estimate = sam_.isam.calculateEstimate();

        //sam_.initial_estimate = new_estimate;
        //current_pose = sam_.initial_estimate.at<Pose3>(Symbol('x',num_of_frame-1)); // pull the last pose out using the size the R matrix;
        //cout<<"GTSAM pose:"<<endl;
        //cout<<current_pose<<endl;
        //cout<<"Total number of new estimate:"<<new_estimate.size()<<endl;
        //cout<<"Total number of History estimate:"<<sam_.initial_estimate_history.size()<<endl;
        //sam_.initial_estimate_history.insert(sam_.initial_estimate);
        //sam_.initial_estimate_history=new_estimate;
        sam_.graph.resize(0);
        sam_.initial_estimate.clear();
    }
    VisualizeLandmarks(image_l, image_r, active_landmarks);

    cout<<"Total number of initial estimate:"<<sam_.initial_estimate.size()<<endl;

}





void VisualOdometry::VisualizeLandmarks(cv::Mat image_l, cv::Mat image_r,
                                        std::vector <Landmark> lm)
{
    using namespace cv;
    if (VISUALIZATION) {
        std::vector <cv::KeyPoint> inliers1, inliers2;
    std::vector <cv::DMatch> good_matches;
    for (unsigned int i=0; i<lm.size(); i++){
        int new_i = static_cast<int>(inliers1.size());
        inliers1.push_back(lm[i].keypoint_l);
        inliers2.push_back(lm[i].keypoint_r);
        good_matches.push_back(cv::DMatch(new_i, new_i, 0));
    }

    Mat res;
    drawMatches(image_l, inliers1, image_r, inliers2, good_matches, res);
    namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// create a window for display.
    imshow( "Display window", res );
    waitKey(0);
    }
}

void VisualOdometry::VisualizeLandmarks(cv::Mat image_l, cv::Mat image_r,
                                        std::vector <cv::KeyPoint> inliers1, std::vector <cv::KeyPoint> inliers2)
{
    using namespace cv;
    if (VISUALIZATION) {
        std::vector <cv::DMatch> good_matches;
    for (unsigned int i=0; i<inliers1.size(); i++){
        good_matches.push_back(cv::DMatch(i, i, 0));
    }

    Mat res;
    drawMatches(image_l, inliers1, image_r, inliers2, good_matches, res);
    namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// create a window for display.
    imshow( "Display window", res );
    waitKey(0);
    }
}

void VisualOdometry::PrintTrajectory(char* filename)
{
    std::ofstream myfile;
    myfile.open (filename);

    for (unsigned int i=0; i<camera_t.size(); i++) {
        myfile << camera_t[i].at<float>(0) << " "
               << camera_t[i].at<float>(1) << " "
               << camera_t[i].at<float>(2) << " " << std::endl;
    }

    myfile.close();
}
