// ROS 2 messages <-> libdogperc (deploy/cpp): the inputs into the ROS-free structs of dogperc/online.hpp
// (no copy of the point cloud), the outputs into ObjectArray, markers and the debug cloud.
#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <dog_perception_msgs/msg/object_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#ifdef DOGPERC_HAVE_KYGI
#include <kygi665_msgs/msg/inspvax.hpp>
#endif

#include <string>

#include "dogperc/online.hpp"

namespace dogperc::ros {

double seconds(const builtin_interfaces::msg::Time& t);   // sec + nanosec * 1e-9, as the Python node
builtin_interfaces::msg::Time stamp(double seconds);

// points into m.data: m must outlive the view
CloudView cloud_view(const sensor_msgs::msg::PointCloud2& m);
Vec3 angular_velocity(const sensor_msgs::msg::Imu& m);
OdomMsg odom(const nav_msgs::msg::Odometry& m);
#ifdef DOGPERC_HAVE_KYGI
InsSample ins_sample(const kygi665_msgs::msg::INSPVAX& m);
#endif

// the planner's message: objects in the det frame of the scan (local_frame), its pose in world_frame
dog_perception_msgs::msg::ObjectArray object_array(const PerceptionPipeline& pipe, const PipelineOutput& out,
                                                   const std::string& local_frame, const std::string& world_frame);
// RViz, in world_frame as the Python node draws them: boxes, ids, and one line per predicted mode
visualization_msgs::msg::MarkerArray track_markers(const PipelineOutput& out, const std::string& world_frame);
visualization_msgs::msg::MarkerArray prediction_markers(const PipelineOutput& out, const std::string& world_frame);
// the accumulated det-frame cloud the detector saw (x, y, z, intensity, dt)
sensor_msgs::msg::PointCloud2 det_cloud(const PipelineOutput& out, const std::string& local_frame);

}  // namespace dogperc::ros
