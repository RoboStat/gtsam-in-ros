cmake_minimum_required(VERSION 2.8.3)
project(visualSLAM)

#find_package(catkin REQUIRED COMPONENTS
#  roscpp
#)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS
	thread
	system
)
find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

include_directories(include
#  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
	#${EIGEN3_INCLUDE_DIR}

)

add_executable(landmark_track
	main.cpp
	src/Landmark.cpp
	src/VisualOdometry.cpp)

target_link_libraries(landmark_track
#	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	${Boost_LIBRARIES}
  tbb
  tbbmalloc
  gtsam
)

# add_dependencies(landmark_track slam_main_generate_messages_cpp)
