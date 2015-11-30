#ifndef VISUALIZATION_H
#define VISUALIZATION_H


#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace gtsam;

enum Color {RED,BLUE,GREEN};
void drawColor(visualization_msgs::Marker &points,
  const std::vector<geometry_msgs::Point> &pose,
  std::string ns,
  Color color)
{
  points.header.frame_id = "/my_frame";
  points.header.stamp =ros::Time::now();
  points.ns= ns;
  points.action = visualization_msgs::Marker::ADD;

  points.id = 0;
  points.type = visualization_msgs::Marker::POINTS;

  switch(color)
  {
    case (RED):
      points.color.r=1.0f;
      break;
    case (BLUE):
      points.color.b=1.0f;
      break;
    case (GREEN):
      points.color.g = 1.0f;
      break;
    }

  points.color.a = 1.0;

  points.scale.x = 0.1;
  points.scale.y = 0.1;
  points.scale.z = 0.1;

  points.points.clear();

  for(auto p=pose.begin(); p!=pose.end(); ++p)
  {
        points.points.push_back(*p);
  }



}






#endif
