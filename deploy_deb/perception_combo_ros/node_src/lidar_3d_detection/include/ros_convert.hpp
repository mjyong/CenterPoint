// ROS 2 messages <-> liblidar3d (deploy/cpp): the inputs into the ROS-free structs of lidar3d/online.hpp
// (no copy of the point cloud), the outputs into ObjectArray, markers and the debug cloud.
#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <perception_interface/msg/lidar_object_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#ifdef LIDAR3D_HAVE_KYGI
#include <kygi665_msgs/msg/inspvax.hpp>
#endif

#include <string>

#include "lidar3d/online.hpp"

namespace lidar3d::ros {

double seconds(const builtin_interfaces::msg::Time& t);   // sec + nanosec * 1e-9, as the Python node
builtin_interfaces::msg::Time stamp(double seconds);

// points into m.data: m must outlive the view
CloudView cloud_view(const sensor_msgs::msg::PointCloud2& m);
Vec3 angular_velocity(const sensor_msgs::msg::Imu& m);
OdomMsg odom(const nav_msgs::msg::Odometry& m);
#ifdef LIDAR3D_HAVE_KYGI
InsSample ins_sample(const kygi665_msgs::msg::INSPVAX& m);
#endif

// the planner's message: objects in the det frame of the scan (local_frame), its pose in world_frame
perception_interface::msg::LidarObjectArray object_array(const PerceptionPipeline& pipe, const PipelineOutput& out,
                                                   const std::string& local_frame, const std::string& world_frame);
// RViz, in world_frame as the Python node draws them: boxes, ids, and one line per predicted mode
visualization_msgs::msg::MarkerArray track_markers(const PipelineOutput& out, const std::string& world_frame);
visualization_msgs::msg::MarkerArray prediction_markers(const PipelineOutput& out, const std::string& world_frame);
// the accumulated det-frame cloud the detector saw (x, y, z, intensity, dt)
sensor_msgs::msg::PointCloud2 det_cloud(const PipelineOutput& out, const std::string& local_frame);
// TF for viewers (the node's publish_tf): world_frame -> local_frame (T_world_det at the scan stamp), and
// world_frame -> the lidar's frame at the cloud's header stamp (the pose there, times T_body_lidar), so that the
// raw cloud, the det cloud, the objects and the world-frame markers show together. None for the lidar when the
// cloud has no frame_id or no pose covers its stamp.
std::vector<geometry_msgs::msg::TransformStamped> viewer_transforms(const PerceptionPipeline& pipe,
                                                                    const PipelineOutput& out,
                                                                    const sensor_msgs::msg::PointCloud2& cloud,
                                                                    const std::string& local_frame,
                                                                    const std::string& world_frame);

}  // namespace lidar3d::ros
