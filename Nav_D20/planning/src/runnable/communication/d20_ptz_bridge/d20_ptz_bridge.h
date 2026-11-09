#pragma once

#include "nav2_interface/action/ptz_camera.hpp"
#include "rte_interface/msg/current_ptz.hpp"
#include "src/runnable/communication/communication_base.h"
#include <geometry_msgs/msg/vector3.hpp>
#include <nav2_interface/msg/nav_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace planning {
namespace communication {
using std::placeholders::_1;
using std::placeholders::_2;
class D20PtzBridge : public CommunicationBase {
public:
  D20PtzBridge(const CommunicationConf::Conf &conf);
  ~D20PtzBridge() override;
  bool Init(PlanningEntry *planning_entry) override;
  bool Proc() override;
  bool Publish() override;
  void PublishPtzCmd(const Eigen::Vector3d &ptz_cmd);

private:
  using PtzCameraAction = nav2_interface::action::PTZCamera;
  using GoalHandlePtzCamera = rclcpp_action::ServerGoalHandle<PtzCameraAction>;
  rclcpp_action::GoalResponse
  handle_ptz_camera_goal(const rclcpp_action::GoalUUID &uuid,
                         std::shared_ptr<const PtzCameraAction::Goal> goal);
  rclcpp_action::CancelResponse handle_ptz_camera_cancel(
      const std::shared_ptr<GoalHandlePtzCamera> goal_handle);
  void handle_ptz_camera_accepted(
      const std::shared_ptr<GoalHandlePtzCamera> goal_handle);

  void ptz_state_callback(const rte_interface::msg::CurrentPTZ::SharedPtr msg);
  void UpdateCameraTF();
  bool CalcPtzCmd();
  void VisualCameraFocusPoint();

private:
  CommunicationConf::D20PtzBridgeConf config_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<rte_interface::msg::CurrentPTZ>::SharedPtr
      sub_ptz_state_;
  rclcpp::Publisher<rte_interface::msg::CurrentPTZ>::SharedPtr pub_ptz_cmd_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      pub_focus_point_;

  rclcpp_action::Server<PtzCameraAction>::SharedPtr ptz_camera_server_;
  std::shared_ptr<GoalHandlePtzCamera> current_ptz_camera_handle_;
  double ptz_state_timestamp_{0.0};
  bool ptz_state_timeout_{false};
  enum class ExecState {
    Idle = 0,
    Executing,
    Failed,
    Succeeded,
    Cancelled,
  };
  struct ExecInfo {
    ExecState exec_state{ExecState::Idle};
    Eigen::Vector3d ptz_cmd{0.0, 0.0, 0.0};
    Eigen::Vector3d ptz_state{0.0, 0.0, 0.0};
    double exec_start_time{0.0};
    double exec_hold_time{0.0};
    double exec_end_time{0.0};
    int sensor_id{0};
    Eigen::Vector3d target_point{0.0, 0.0, 0.0};
    Eigen::Vector3d target_scale{1.0, 1.0, 1.0};
    double target_image_scale{0.5};
  };
  ExecInfo exec_info_;
};

} // namespace communication
} // namespace planning
