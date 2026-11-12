/*
 * @Author: tian.zhao tian.zhao@seres.cn
 * @Date: 2024-03-08 17:20:48
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2025-01-03 13:58:35
 */
#pragma once
#include "geometry_msgs/msg/pose2_d.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "src/common/structures.h"
#include "src/common/util/config_helper.h"
#include "src/common/util/time_keeper.h"
#include "tf2/buffer_core.h"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
using planning::common::util::ConfigHelper;
using planning::common::util::TimeKeeper;

using geometry_msgs::msg::Point;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::Pose2D;
using geometry_msgs::msg::Quaternion;
using geometry_msgs::msg::Vector3;
using std_msgs::msg::ColorRGBA;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

// Seconds a marker stays up after it was received. It was 0.05, one 20 Hz cycle, so in Foxglove playback the
// markers blinked and a paused frame mostly had none. Arrays whose marker count varies start with a DELETEALL
// (appendDeleteAll); this is the fallback when planning stops publishing (2026-11-09).
#ifndef LIFE_TIME
#define LIFE_TIME 0.5
#endif

class VisualizationUtil {
public:
  // clang-format off
  enum colors {
    BLACK = 0,  BROWN = 1,      BLUE = 2,     CYAN = 3,     GREY = 4,       DARK_GREY = 5,
    GREEN = 6,  LIME_GREEN = 7, MAGENTA = 8,  ORANGE = 9,   PURPLE = 10,
    RED = 11,   PINK = 12,      WHITE = 13,   YELLOW = 14,  TRANSLUCENT = 15,
    TRANSLUCENT_LIGHT = 16,     TRANSLUCENT_DARK = 17,      CLEAR = 19,     DEFAULT = 20
  };

  static ColorRGBA getColor(colors color, double alpha = 1.0) {
    ColorRGBA result;
    switch (color) {
    case RED:               result.r = 0.8;result.g = 0.1;result.b = 0.1;result.a = alpha;      break;
    case GREEN:             result.r = 0.1;result.g = 0.8;result.b = 0.1;result.a = alpha;      break;
    case BLUE:              result.r = 0.1;result.g = 0.1;result.b = 0.8;result.a = alpha;      break;
    case GREY:              result.r = 0.9;result.g = 0.9;result.b = 0.9;result.a = alpha;      break;
    case DARK_GREY:         result.r = 0.6;result.g = 0.6;result.b = 0.6;result.a = alpha;      break;
    case WHITE:             result.r = 1.0;result.g = 1.0;result.b = 1.0;result.a = alpha;      break;
    case ORANGE:            result.r = 1.0;result.g = 0.5;result.b = 0.0;result.a = alpha;      break;
    case TRANSLUCENT_LIGHT: result.r = 0.1;result.g = 0.1;result.b = 0.1;result.a = 0.1;        break;
    case TRANSLUCENT:       result.r = 0.1;result.g = 0.1;result.b = 0.1;result.a = 0.25;       break;
    case TRANSLUCENT_DARK:  result.r = 0.1;result.g = 0.1;result.b = 0.1;result.a = 0.5;        break;
    case BLACK:             result.r = 0.0;result.g = 0.0;result.b = 0.0;result.a = alpha;      break;
    case YELLOW:            result.r = 1.0;result.g = 1.0;result.b = 0.0;result.a = alpha;      break;
    case BROWN:             result.r = 0.597;result.g = 0.296;result.b = 0.0;result.a = alpha;  break;
    case PINK:              result.r = 1.0;result.g = 0.4;result.b = 1;result.a = alpha;        break;
    case LIME_GREEN:        result.r = 0.6;result.g = 1.0;result.b = 0.2;result.a = alpha;      break;
    case CLEAR:             result.r = 1.0;result.g = 1.0;result.b = 1.0;result.a = 0.0;        break;
    case PURPLE:            result.r = 0.597;result.g = 0.0;result.b = 0.597;result.a = alpha;  break;
    case CYAN:              result.r = 0.0;result.g = 1.0;result.b = 1.0;result.a = alpha;      break;
    case MAGENTA:           result.r = 1.0;result.g = 0.0;result.b = 1.0;result.a = alpha;      break;
    case DEFAULT:
    default:                result.r = 0.1;result.g = 0.1;result.b = 0.8;result.a = alpha;      break;
    }
    return result;
  }

  static ColorRGBA GetColorByGait(const planning::Gait &gait) {
    ColorRGBA result;
    if(gait == planning::Gait::LAY_ON || gait == planning::Gait::DAMPING || gait == planning::Gait::STAND) {
      result = getColor(WHITE);
    } else if (gait == planning::Gait::WALK) {
      result = getColor(GREEN);
    } else if (gait == planning::Gait::CLIMB) {
      result = getColor(ORANGE);
    }
    return result;
  } 

  static Point createMarkerPosition(double x, double y, double z) {
    Point point;  point.x = x;  point.y = y;  point.z = z;  return point;
  }

  static Quaternion createMarkerOrientation(double x, double y, double z,
                                            double w) {
    Quaternion quaternion;
    quaternion.x = x; quaternion.y = y; quaternion.z = z; quaternion.w = w;
    return quaternion;
  }

  static Vector3 createMarkerScale(double x, double y, double z) {
    Vector3 scale;  scale.x = x;  scale.y = y;  scale.z = z;  return scale;
  }

  static ColorRGBA createMarkerColor(double r, double g, double b, double a) {
    ColorRGBA color;
    color.r = r;  color.g = g;  color.b = b;  color.a = a;  return color;
  }
  // clang-format on
  static Point TransToAbsCoord(const Point &point, const Pose &origin) {
    // rotation
    Point rot_p;
    double yaw = tf2::getYaw(origin.orientation);
    rot_p.x = (cos(yaw) * point.x) + ((-1.0) * sin(yaw) * point.y);
    rot_p.y = (sin(yaw) * point.x) + (cos(yaw) * point.y);

    // translation
    Point res;
    res.x = rot_p.x + origin.position.x;
    res.y = rot_p.y + origin.position.y;
    res.z = origin.position.z;

    return res;
  }

  static Pose TransToAbsCoord(const Pose &target, const Pose &origin) {
    Pose res;
    res.position = TransToAbsCoord(target.position, origin);
    double target_yaw = tf2::getYaw(target.orientation);
    double origin_yaw = tf2::getYaw(origin.orientation);
    res.orientation = tf2::toMsg(
        tf2::Quaternion(tf2::Vector3(0, 0, 1), target_yaw + origin_yaw));
    return res;
  }

  static Point TransToRelCoord(const Point &point, const Pose &origin) {
    // translation
    Point trans_p;
    trans_p.x = point.x - origin.position.x;
    trans_p.y = point.y - origin.position.y;

    // rotation (use inverse matrix of rotation)
    double yaw = tf2::getYaw(origin.orientation);
    Point res;
    res.x = (cos(yaw) * trans_p.x) + (sin(yaw) * trans_p.y);
    res.y = ((-1.0) * sin(yaw) * trans_p.x) + (cos(yaw) * trans_p.y);
    res.z = point.z;

    return res;
  }

  static Pose TransToRelCoord(const Pose &target, const Pose &origin) {
    Pose res;
    res.position = TransToRelCoord(target.position, origin);
    double target_yaw = tf2::getYaw(target.orientation);
    double origin_yaw = tf2::getYaw(origin.orientation);
    res.orientation = tf2::toMsg(
        tf2::Quaternion(tf2::Vector3(0, 0, 1), target_yaw - origin_yaw));
    return res;
  }

  static Marker createDefaultMarker(const std::string &frame_id,
                                    const std::string &ns, const int32_t id,
                                    const int32_t type, const Vector3 &scale,
                                    const ColorRGBA &color) {
    Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = rclcpp::Time();
    marker.ns = ns;
    marker.id = id;
    marker.type = type;
    marker.action = Marker::ADD;
    marker.lifetime = rclcpp::Duration::from_seconds(LIFE_TIME);
    marker.pose.position = createMarkerPosition(0.0, 0.0, 0.0);
    marker.pose.orientation = createMarkerOrientation(0.0, 0.0, 0.0, 1.0);
    marker.scale = scale;
    marker.color = color;
    marker.frame_locked = true;
    return marker;
  }

  // A DELETEALL first in the array: RViz and Foxglove drop the topic's earlier markers, then add the rest.
  static void appendDeleteAll(MarkerArray *marker_array) {
    Marker marker;
    marker.action = Marker::DELETEALL;
    marker_array->markers.push_back(marker);
  }

  static void appendMarkerArray(const MarkerArray &additional_marker_array,
                                MarkerArray *marker_array) {
    for (const auto &marker : additional_marker_array.markers) {
      marker_array->markers.push_back(marker);
    }
  }

  static Marker
  createLineStripMarker(const std::string &frame_id, const std::string &ns,
                        int id, const ColorRGBA &color,
                        const std::vector<Eigen::Vector3d> &points) {
    Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = rclcpp::Time();
    marker.ns = ns;
    marker.id = id;
    marker.type = Marker::LINE_STRIP;
    marker.action = Marker::ADD;
    marker.lifetime = rclcpp::Duration::from_seconds(LIFE_TIME);
    for (size_t i = 0; i < points.size(); i++) {
      Point pt;
      pt.x = points.at(i).x();
      pt.y = points.at(i).y();
      pt.z = points.at(i).z();
      marker.points.emplace_back(pt);
    }
    marker.scale = createMarkerScale(0.1, 0.0, 0.0);
    marker.color = color;
    marker.frame_locked = true;
    return marker;
  }

  static Marker
  createLineStripMarker(const std::string &frame_id, const std::string &ns,
                        int id, const ColorRGBA &color,
                        const std::vector<Eigen::Vector2d> &points) {
    std::vector<Eigen::Vector3d> points3d;
    for (const auto &point : points) {
      points3d.emplace_back(point.x(), point.y(), 0.0);
    }
    return createLineStripMarker(frame_id, ns, id, color, points3d);
  }

  static Marker
  createLineStripMarker(const std::string &frame_id, const std::string &ns,
                        int id, const ColorRGBA &color,
                        const std::vector<planning::Pose2d> &poses) {
    std::vector<Eigen::Vector3d> points3d;
    for (const auto &pose : poses) {
      points3d.emplace_back(pose.position.x(), pose.position.y(), 0.0);
    }
    return createLineStripMarker(frame_id, ns, id, color, points3d);
  }

  static Marker
  createLineListMarker(const std::string &frame_id, const std::string &ns,
                       int id, const ColorRGBA &color,
                       const std::vector<Eigen::Vector3d> &points) {

    Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = rclcpp::Time();
    marker.ns = ns;
    marker.id = id;
    marker.type = Marker::LINE_LIST;
    marker.action = Marker::ADD;
    marker.lifetime = rclcpp::Duration::from_seconds(LIFE_TIME);
    for (size_t i = 0; i < points.size(); i++) {
      Point pt;
      pt.x = points.at(i).x();
      pt.y = points.at(i).y();
      pt.z = points.at(i).z();
      marker.points.emplace_back(pt);
    }
    if (points.size() % 2 != 0) {
      Point pt;
      pt.x = points.back().x();
      pt.y = points.back().y();
      pt.z = 0.0;
      marker.points.emplace_back(pt);
    }
    marker.scale = createMarkerScale(0.1, 0.0, 0.0);
    marker.color = color;
    marker.frame_locked = true;
    return marker;
  }

  static Pose transformBaseLink2Center(const Pose &pose,
                                       const double &offset_x) {
    tf2::Transform tf_map2base;
    tf_map2base.setOrigin(
        tf2::Vector3(pose.position.x, pose.position.y, pose.position.z));
    // w at the end in the constructor
    tf_map2base.setRotation(
        tf2::Quaternion(pose.orientation.x, pose.orientation.y,
                        pose.orientation.z, pose.orientation.w));

    tf2::Vector3 base2center;
    base2center.setX(offset_x);
    base2center.setY(0.0);
    base2center.setZ(0.0);
    base2center.setW(1.0);

    const auto map2center = tf_map2base * base2center;
    Pose center_pose;
    center_pose.position.x = map2center.x();
    center_pose.position.y = map2center.y();
    center_pose.position.z = map2center.z();
    center_pose.orientation = pose.orientation;

    return center_pose;
  }

  static Marker createMarkerSphere(const std::string &frame_id,
                                   const std::string &ns, int id,
                                   const Pose &pose, const Vector3 &scale,
                                   const ColorRGBA &color,
                                   const Pose &origin = Pose(),
                                   bool to_relative = false) {
    Marker marker;
    marker.header.frame_id = to_relative ? "base_link" : frame_id;
    marker.header.stamp = rclcpp::Time();
    marker.ns = ns;
    marker.id = id;
    marker.lifetime = rclcpp::Duration::from_seconds(LIFE_TIME);
    marker.type = Marker::SPHERE;
    marker.action = Marker::ADD;
    marker.scale = scale;
    marker.pose = to_relative ? TransToRelCoord(pose, origin) : pose;
    marker.color = color;
    return marker;
  }

  static Marker createMarkerCube(const std::string &frame_id,
                                 const std::string &ns, int id,
                                 const Pose &pose, const Vector3 &scale,
                                 const ColorRGBA &color,
                                 const Pose &origin = Pose(),
                                 bool to_relative = false) {
    Marker marker;
    marker.header.frame_id = to_relative ? "base_link" : frame_id;
    marker.header.stamp = rclcpp::Time();
    marker.ns = ns;
    marker.id = id;
    marker.lifetime = rclcpp::Duration::from_seconds(LIFE_TIME);
    marker.type = Marker::CUBE;
    marker.action = Marker::ADD;
    marker.scale = scale;
    marker.pose = to_relative ? TransToRelCoord(pose, origin) : pose;
    marker.color = color;
    return marker;
  }

  static Marker createMarkerArrow(const std::string &frame_id,
                                  const std::string &ns, int id,
                                  const Pose &pose, const Vector3 &scale,
                                  const ColorRGBA &color) {
    Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = rclcpp::Time();
    marker.ns = ns;
    marker.id = id;
    marker.lifetime = rclcpp::Duration::from_seconds(LIFE_TIME);
    marker.type = Marker::ARROW;
    marker.action = Marker::ADD;
    marker.scale = scale;
    marker.pose = pose;
    marker.color = color;
    return marker;
  }

  static Marker createMarkerText(const std::string &frame_id,
                                 const std::string &ns, int id,
                                 const Pose &pose, const std::string &text) {
    Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = rclcpp::Time();
    marker.ns = ns;
    marker.id = id;
    marker.type = Marker::TEXT_VIEW_FACING;
    marker.action = Marker::ADD;
    marker.lifetime = rclcpp::Duration::from_seconds(LIFE_TIME);
    marker.pose = pose;
    marker.scale = createMarkerScale(0.0, 0.0, 1.0);
    marker.color = createMarkerColor(1.0, 1.0, 1.0, 1.0);
    marker.text = text;
    return marker;
  }

  static Marker createMarkerFootprint(const std::string &frame_id,
                                      const std::string &ns, int id,
                                      const Pose2D &pose2d,
                                      const ColorRGBA &color) {
    Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = rclcpp::Time();
    marker.ns = ns;
    marker.id = id;
    marker.type = Marker::LINE_STRIP;
    marker.action = Marker::ADD;
    marker.lifetime = rclcpp::Duration::from_seconds(LIFE_TIME);
    marker.points = getFootprint(pose2d.x, pose2d.y, pose2d.theta);
    marker.scale = createMarkerScale(0.02, 0.0, 0.0);
    marker.color = color;
    return marker;
  }

  static std::vector<geometry_msgs::msg::Point> getFootprint(double x, double y,
                                                             double yaw) {
    using Eigen::MatrixXd;
    const double robot_l = ConfigHelper::GetConfig().robot_param().length();
    const double robot_w = ConfigHelper::GetConfig().robot_param().width();

    MatrixXd robot_outline(2, 5);
    robot_outline << -robot_l / 2, robot_l / 2, robot_l / 2, -robot_l / 2,
        -robot_l / 2, robot_w / 2, robot_w / 2, -robot_w / 2, -robot_w / 2,
        robot_w / 2;

    Eigen::Matrix2d rot_yaw;
    rot_yaw << std::cos(yaw), -std::sin(yaw), std::sin(yaw), std::cos(yaw);

    robot_outline = rot_yaw * robot_outline;
    robot_outline.row(0).array() += x;
    robot_outline.row(1).array() += y;
    std::vector<geometry_msgs::msg::Point> out_shapes;
    for (int i = 0; i < robot_outline.cols(); ++i) {
      geometry_msgs::msg::Point p;
      p.x = robot_outline.col(i).x();
      p.y = robot_outline.col(i).y();
      p.z = 0.0;
      out_shapes.emplace_back(p);
    }
    return out_shapes;
  }

  static MarkerArray createMarkerFootprintCircle(const std::string &frame_id,
                                                 const std::string &ns, int id,
                                                 const Pose2D &pose2d,
                                                 const ColorRGBA &color) {
    MarkerArray markers;
    auto circles = ConfigHelper::Instance()->GetBoundingCircles(
        pose2d.x, pose2d.y, pose2d.theta);
    for (auto &circle : circles) {
      Marker marker;
      marker.header.frame_id = frame_id;
      marker.header.stamp = rclcpp::Time();
      marker.ns = ns;
      marker.id = id++;
      marker.type = Marker::SPHERE;
      marker.action = Marker::ADD;
      marker.lifetime = rclcpp::Duration::from_seconds(LIFE_TIME);
      marker.color = color;
      marker.color.a = 0.3;
      marker.pose.position.x = circle.center_x;
      marker.pose.position.y = circle.center_y;
      marker.pose.position.z = 0.0;
      marker.pose.orientation.w = 1;
      double scale = circle.radius * 2.0;
      marker.scale = createMarkerScale(scale, scale, 0.1);
      markers.markers.emplace_back(marker);
    }
    auto circle = ConfigHelper::Instance()->GetBoundingCircle(
        pose2d.x, pose2d.y, pose2d.theta);
    Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = rclcpp::Time();
    marker.ns = ns;
    marker.id = id++;
    marker.type = Marker::SPHERE;
    marker.action = Marker::ADD;
    marker.lifetime = rclcpp::Duration::from_seconds(LIFE_TIME);
    marker.color = color;
    marker.color.a = 0.2;
    marker.pose.position.x = circle.center_x;
    marker.pose.position.y = circle.center_y;
    marker.pose.position.z = 0.0;
    marker.pose.orientation.w = 1;
    double scale = circle.radius * 2.0;
    marker.scale = createMarkerScale(scale, scale, 0.1);
    markers.markers.emplace_back(marker);
    return markers;
  }
};
