/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-23 16:05:32
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2025-09-12 17:46:29
 */
#pragma once
#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/intersects.hpp>
#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <cstddef>
#define EIGEN_MPL2_ONLY
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace planning {

struct Point2d : public Eigen::Vector2d {
  Point2d() : Eigen::Vector2d(0, 0) {}
  Point2d(const double x, const double y) : Eigen::Vector2d(x, y) {}
  Point2d(const Eigen::Vector2d &v) : Eigen::Vector2d(v) {}

  std::string toString() const {
    return std::to_string(x()) + ", " + std::to_string(y());
  }
};
using Segment2d = boost::geometry::model::segment<Point2d>;
using Box2d = boost::geometry::model::box<Point2d>;
using LineString2d = boost::geometry::model::linestring<Point2d>;
using LinearRing2d = boost::geometry::model::ring<Point2d>;
using Polygon2d = boost::geometry::model::polygon<Point2d>;
using MultiPoint2d = boost::geometry::model::multi_point<Point2d>;
using MultiLineString2d =
    boost::geometry::model::multi_linestring<LineString2d>;
using MultiPolygon2d = boost::geometry::model::multi_polygon<Polygon2d>;

struct Point3d : public Eigen::Vector3d {
  Point3d() : Eigen::Vector3d(0, 0, 0) {}
  Point3d(const double x, const double y, const double z)
      : Eigen::Vector3d(x, y, z) {}
  Point3d(const Eigen::Vector3d &v) : Eigen::Vector3d(v) {}

  std::string toString() const {
    return std::to_string(x()) + ", " + std::to_string(y()) + ", " +
           std::to_string(z());
  }
};
using Segment3d = boost::geometry::model::segment<Point3d>;
using Box3d = boost::geometry::model::box<Point3d>;
using LineString3d = boost::geometry::model::linestring<Point3d>;
using LinearRing3d = boost::geometry::model::ring<Point3d>;
using Polygon3d = boost::geometry::model::polygon<Point3d>;
using MultiPoint3d = boost::geometry::model::multi_point<Point3d>;
using MultiLineString3d =
    boost::geometry::model::multi_linestring<LineString3d>;
using MultiPolygon3d = boost::geometry::model::multi_polygon<Polygon3d>;

using Quaternion = Eigen::Quaterniond;
using Transform2d = Eigen::Transform<double, 2, Eigen::Affine>;
using Transform3d = Eigen::Transform<double, 3, Eigen::Affine>;

struct TransformStamped {
  double timestamp;
  std::string frame_id;
  std::string child_frame_id;
  Transform3d transform;
  TransformStamped() = default;
  TransformStamped(const double timestamp, const std::string &frame_id,
                   const std::string &child_frame_id,
                   const Transform3d &transform)
      : timestamp(timestamp), frame_id(frame_id),
        child_frame_id(child_frame_id), transform(transform) {}
  TransformStamped(const double timestamp, const std::string &frame_id,
                   const std::string &child_frame_id,
                   const Eigen::Vector3d &translation,
                   const Eigen::Quaterniond &rotation)
      : timestamp(timestamp), frame_id(frame_id),
        child_frame_id(child_frame_id) {
    transform = Transform3d::Identity();
    transform.translate(translation);
    transform.rotate(rotation);
  }
};

struct Pose2d {
  Point2d position;
  double theta;
  Pose2d() {
    position = Point2d(0, 0);
    theta = 0;
  }
  Pose2d(const Point2d &position, const double theta)
      : position(position), theta(theta) {}
};

struct Pose3d {
  Point3d position;
  Quaternion orientation;
  Pose3d() {
    position = Point3d(0, 0, 0);
    orientation = Quaternion::Identity();
  }
  Pose3d(const Point3d &position, const Quaternion &orientation)
      : position(position), orientation(orientation) {}
  Eigen::Vector3d euler() const {
    const auto &q = orientation;
    double roll = 0;
    double pitch = 0;
    double yaw = 0;
    double sqw;
    double sqx;
    double sqy;
    double sqz;

    sqx = q.x() * q.x();
    sqy = q.y() * q.y();
    sqz = q.z() * q.z();
    sqw = q.w() * q.w();

    // Cases derived from https://orbitalstation.wordpress.com/tag/quaternion/
    double sarg =
        -2 * (q.x() * q.z() - q.w() * q.y()) /
        (sqx + sqy + sqz + sqw); /* normalization added from urdfom_headers */
    if (sarg <= -0.99999) {
      pitch = -0.5 * M_PI;
      roll = 0;
      yaw = -2 * atan2(q.y(), q.x());
    } else if (sarg >= 0.99999) {
      pitch = 0.5 * M_PI;
      roll = 0;
      yaw = 2 * atan2(q.y(), q.x());
    } else {
      pitch = asin(sarg);
      roll = atan2(2 * (q.y() * q.z() + q.w() * q.x()), sqw - sqx - sqy + sqz);
      yaw = atan2(2 * (q.x() * q.y() + q.w() * q.z()), sqw + sqx - sqy - sqz);
    }
    return Eigen::Vector3d(roll, pitch, yaw);
  }
};

struct Point2dStamped {
  Point2d point;
  double timestamp;
  Point2dStamped() {
    point = Point2d(0, 0);
    timestamp = 0;
  }
  Point2dStamped(const Point2d &point, const double timestamp)
      : point(point), timestamp(timestamp) {}
};
struct Point3dStamped {
  Point3d point;
  double timestamp;
  Point3dStamped() {
    point = Point3d(0, 0, 0);
    timestamp = 0;
  }
  Point3dStamped(const Point3d &point, const double timestamp)
      : point(point), timestamp(timestamp) {}
};

struct Pose2dStamped {
  Pose2d pose;
  double timestamp;
  Pose2dStamped() {
    pose = Pose2d();
    timestamp = 0;
  }
  Pose2dStamped(const Pose2d &pose, const double timestamp)
      : pose(pose), timestamp(timestamp) {}
};
struct Pose3dStamped {
  Pose3d pose;
  double timestamp;
  Pose3dStamped() {
    pose = Pose3d();
    timestamp = 0;
  }
  Pose3dStamped(const Pose3d &pose, const double timestamp)
      : pose(pose), timestamp(timestamp) {}
};

struct Twist {
  Eigen::Vector3d linear;
  Eigen::Vector3d angular;
  Twist() {
    linear = Eigen::Vector3d::Zero();
    angular = Eigen::Vector3d::Zero();
  }
  Twist(const Eigen::Vector3d &linear, const Eigen::Vector3d &angular)
      : linear(linear), angular(angular) {}
  void reset() {
    linear.setZero();
    angular.setZero();
  }
};

struct TwistStamped {
  Twist twist;
  double timestamp;
  TwistStamped() {
    twist = Twist();
    timestamp = 0;
  }
  TwistStamped(const Twist &twist, const double timestamp)
      : twist(twist), timestamp(timestamp) {}
};

struct Odometry {
  double timestamp;
  Pose3d pose;
  Twist twist;
  Odometry() {
    timestamp = 0;
    pose = Pose3d();
    twist = Twist();
  }
};

enum class Gait { LAY_ON = 2, DAMPING = 3, STAND = 4, WALK = 10, CLIMB = 11 };

} // namespace planning

BOOST_GEOMETRY_REGISTER_POINT_2D(                            // NOLINT
    planning::Point2d, double, cs::cartesian, x(), y())      // NOLINT
BOOST_GEOMETRY_REGISTER_POINT_3D(                            // NOLINT
    planning::Point3d, double, cs::cartesian, x(), y(), z()) // NOLINT
