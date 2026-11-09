#pragma once

#include "mc_interface/msg/joints_control.hpp"
#include "mc_interface/msg/joints_state.hpp"
#include "perception_interface/msg/objects.hpp"
#include "src/common/util/time_keeper.h"
#include "src/runnable/communication/communication_base.h"
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
namespace planning {
namespace communication {

class R11HeadBridge : public CommunicationBase {
public:
  R11HeadBridge(const CommunicationConf::Conf &conf);
  ~R11HeadBridge() override;
  bool Init(PlanningEntry *planning_entry) override;
  bool Proc() override;
  bool Publish() override;

  void set_target_img_position(const double x, const double y, const double w,
                               const double h, const double timestamp) {
    object_info_.img_position.x() = x;
    object_info_.img_position.y() = y;
    object_info_.img_size.x() = w;
    object_info_.img_size.y() = h;
    object_info_.timestamp = timestamp;
  }

private:
  void head_state_callback(const mc_interface::msg::JointsState::SharedPtr msg);
  void
  test_object_callback(const geometry_msgs::msg::Quaternion::SharedPtr msg);
  void CalcTrackingCmd();

private:
  CommunicationConf::R11HeadBridgeConf config_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<mc_interface::msg::JointsState>::SharedPtr
      sub_head_state_;
  rclcpp::Publisher<mc_interface::msg::JointsControl>::SharedPtr pub_head_cmd_;
  rclcpp::Subscription<geometry_msgs::msg::Quaternion>::SharedPtr
      sub_test_object_;
  enum class HeadMotorID {
    PITCH = 23,
    ROLL = 24,
    YAW = 22,
  };

  struct MotorParam {
    double max_position;
    double min_position;
    double kp;
    double kd;
    MotorParam()
        : max_position(0.5236), min_position(-0.5236), kp(10.0), kd(3.0) {}
  };
  struct MotorInfo {
    double timestamp;
    double q;
    double qd;
    double tau;
    double kp;
    double kd;
    MotorInfo() : timestamp(0.0), q(0.0), qd(0.0), tau(0.0), kp(0.0), kd(0.0) {}
  };
  struct JointInfo {
    MotorParam param;
    MotorInfo feedback;
    MotorInfo command;
    JointInfo() : param(), feedback(), command() {}
    void command_clamp() {
      command.q = std::clamp(command.q, param.min_position, param.max_position);
      command.qd = 0;
      command.tau = 0;
    }
  };
  std::unordered_map<HeadMotorID, JointInfo> joint_infos_;
  struct ObjectInfo {
    double timestamp;
    Eigen::Vector2d position;
    Eigen::Vector2d img_position;
    Eigen::Vector2d img_size;
    ObjectInfo()
        : timestamp(0.0), position(Eigen::Vector2d::Zero()),
          img_position(Eigen::Vector2d::Zero()),
          img_size(Eigen::Vector2d::Zero()) {}
  };
  ObjectInfo object_info_;
  double lpf_img_x_ = 0.5;
  double lpf_img_y_ = 0.5;
};

} // namespace communication
} // namespace planning
