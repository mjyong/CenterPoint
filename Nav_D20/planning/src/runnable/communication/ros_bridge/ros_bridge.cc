#include "src/runnable/communication/ros_bridge/ros_bridge.h"
#include <cstdlib>
#include "src/common/util/config_helper.h"
#include "src/runnable/entry/planning_entry.h"
#include "src/runnable/mission/recharging/recharging.h"
#include "src/runnable/mission/tracking/tracking.h"
using planning::common::util::ConfigHelper;

namespace planning {
namespace communication {

// Bag 回放时消息时间戳和本机时钟差几周，按戳做超时会一直暂停（T48：暂停以后再也没恢复）。
// 只有调试脚本设置 PLANNING_TIMEOUT_WALL_CLOCK=1 时，改用收到消息的时刻；狗上不设，行为不变。
static double TimeoutReference(double message_stamp) {
  const char *flag = std::getenv("PLANNING_TIMEOUT_WALL_CLOCK");
  if (flag != nullptr && flag[0] == '1' && flag[1] == '\0') {
    return planning::common::util::TimeKeeper::TimeNowSec();
  }
  return message_stamp;
}

RosBridge::RosBridge(const CommunicationConf::Conf &conf) : rate_(20) {
  conf_ = conf;
  config_ = conf_.ros_bridge_conf();
}
RosBridge::~RosBridge() {}

bool RosBridge::ok() { return rclcpp::ok(); }

bool RosBridge::Init(PlanningEntry *planning_entry) {
  planning_entry_ = planning_entry;
  node_ = std::make_shared<rclcpp::Node>("planning");
  diagnostic_bridge_ = std::make_unique<DiagnosticBridge>(node_);

  if (config_.has_sub_odom()) {
    odom_sub_ = std::make_unique<OdomSub>(
        node_, config_.sub_odom().name(),
        rclcpp::QoS(config_.sub_odom().queue_size()).get_rmw_qos_profile());
    if (config_.sub_odom().has_timeout()) {
      timeout_states_[config_.sub_odom().name()] = {
          0.0, true, config_.sub_odom().timeout(), [this]() {
            auto entry = planning_entry_->mission_entry_ptr();
            return entry->GetCurrentMissionType() == MissionConf::Tracking;
          }};
    }
  }

  if (config_.has_sub_occupancy_grid()) {
    occ_sub_ = std::make_unique<OccSub>(
        node_, config_.sub_occupancy_grid().name(),
        rclcpp::QoS(config_.sub_occupancy_grid().queue_size())
            .get_rmw_qos_profile());
    if (config_.sub_occupancy_grid().has_timeout()) {
      timeout_states_[config_.sub_occupancy_grid().name()] = {
          0.0, true, config_.sub_occupancy_grid().timeout(), [this]() {
            auto entry = planning_entry_->mission_entry_ptr();
            return entry->GetCurrentMissionType() == MissionConf::Tracking;
          }};
    }
  }

  if (config_.sync_odom_occ() && odom_sub_ && occ_sub_) {
    sync_ =
        std::make_unique<Synchronizer>(SyncPolicy(10), *odom_sub_, *occ_sub_);
    sync_->registerCallback(&RosBridge::sync_callback, this);
  } else {
    if (odom_sub_)
      odom_sub_->registerCallback(&RosBridge::odom_callback, this);
    if (occ_sub_)
      occ_sub_->registerCallback(&RosBridge::occupancy_callback, this);
  }

  if (config_.has_sub_global_occupancy_grid()) {
    sub_global_occ_ = node_->create_subscription<nav_msgs::msg::OccupancyGrid>(
        config_.sub_global_occupancy_grid().name(),
        rclcpp::QoS(1).reliable().durability(
            rclcpp::DurabilityPolicy::TransientLocal),
        std::bind(&RosBridge::global_occupancy_callback, this,
                  std::placeholders::_1));
  }

  if (config_.has_sub_map_switch()) {
    sub_map_switch_ = node_->create_subscription<std_msgs::msg::String>(
        config_.sub_map_switch().name(),
        rclcpp::QoS(1).reliable().transient_local().keep_last(1),
        std::bind(&RosBridge::map_switch_callback, this,
                  std::placeholders::_1));
  }
  if (config_.has_sub_map_update()) {
    sub_map_update_ =
        node_->create_subscription<common_interface::msg::PathFile>(
            config_.sub_map_update().name(),
            config_.sub_map_update().queue_size(),
            std::bind(&RosBridge::map_update_callback, this,
                      std::placeholders::_1));
  }

  if (config_.has_sub_joint_state()) {
    sub_joint_state_ = node_->create_subscription<sensor_msgs::msg::JointState>(
        config_.sub_joint_state().name(),
        config_.sub_joint_state().queue_size(),
        std::bind(&RosBridge::joint_state_callback, this, _1));
    if (config_.sub_joint_state().has_timeout()) {
      timeout_states_[config_.sub_joint_state().name()] = {
          0.0, true, config_.sub_joint_state().timeout()};
    }
  }

  if (config_.has_sub_octomap()) {
    sub_local_octomap_ = node_->create_subscription<octomap_msgs::msg::Octomap>(
        config_.sub_octomap().name(), config_.sub_octomap().queue_size(),
        std::bind(&RosBridge::local_octomap_callback, this, _1));
    if (config_.sub_octomap().has_timeout()) {
      timeout_states_[config_.sub_octomap().name()] = {
          0.0, true, config_.sub_octomap().timeout()};
    }
  }

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
  tf_listener_ =
      std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, node_, true);

  if (config_.has_pub_cmd_vel()) {
    pub_cmd_ = node_->create_publisher<geometry_msgs::msg::Twist>(
        config_.pub_cmd_vel().name(), config_.pub_cmd_vel().queue_size());
  }

  if (config_.has_pub_motion()) {
    pub_motion_ = node_->create_publisher<mc_interface::msg::Motion>(
        config_.pub_motion().name(), config_.pub_motion().queue_size());
  }

  tracking_bridge_ =
      std::make_unique<TrackingBridge>(node_, planning_entry_, config_);
  recharging_bridge_ = std::make_unique<RechargingBridge>(
      node_, planning_entry_, config_, timeout_states_);
  following_bridge_ = std::make_unique<FollowingBridge>(
      node_, planning_entry_, config_, timeout_states_);
  skill_bridge_ = std::make_unique<SkillBridge>(
      node_, planning_entry_, config_, timeout_states_);

  odometry_.pose.position.x() = 1;
  odometry_.pose.position.y() = -1;

  if (config_.has_sub_init_pose()) {
    init_pose_sub_ = node_->create_subscription<
        geometry_msgs::msg::PoseWithCovarianceStamped>(
        config_.sub_init_pose().name(), config_.sub_init_pose().queue_size(),
        std::bind(&RosBridge::OnInitPose, this, std::placeholders::_1));
  }
  if (config_.has_sub_goal_pose()) {
    goal_pose_sub_ =
        node_->create_subscription<geometry_msgs::msg::PoseStamped>(
            config_.sub_goal_pose().name(),
            config_.sub_goal_pose().queue_size(),
            std::bind(&RosBridge::OnGoalPose, this, std::placeholders::_1));
  }

  TransformStamped tf(odometry_.timestamp, "map", "base_link",
                      odometry_.pose.position, odometry_.pose.orientation);
  TfBuffer::Instance()->add(tf);
  return true;
}

bool RosBridge::Proc() {
  bool necessary_input_timeout =
      config_.enable_timeout_check() ? checkTimeout() : false;
  auto mission_entry = planning_entry_->mission_entry_ptr();
  DebugDatas::Instance()->emplace(
      "mission_type", static_cast<int>(mission_entry->GetCurrentMissionType()));
  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle) {
    auto current_mission = mission_entry->GetCurrentMissionPtr();
    DebugDatas::Instance()->emplace(
        "mission_status", static_cast<int>(current_mission->mission_status()));
    double time_now = TimeKeeper::TimeNowSec();
    if (current_mission->mission_status() ==
            MissionFSMBase::StateType::active ||
        current_mission->mission_status() ==
            MissionFSMBase::StateType::standstill) {
      if (necessary_input_timeout && !is_timeout_paused_) {
        if (timeout_pause_start_time_ < 0.1) {
          timeout_pause_start_time_ = time_now;
        } else if ((time_now - timeout_pause_start_time_) >=
                   config_.timeout_pause_duration()) {
          AWARN << "Data timeout exceeded, pausing mission";
          current_mission->pause();
          is_timeout_paused_ = true;
        }
      } else if (!necessary_input_timeout && is_timeout_paused_) {
        if (timeout_resume_start_time_ < 0.1) {
          timeout_resume_start_time_ = time_now;
        } else if ((time_now - timeout_resume_start_time_) >=
                   config_.timeout_resume_duration()) {
          AWARN << "Data recovered, resuming mission";
          current_mission->resume();
          is_timeout_paused_ = false;
        } else {
          AWARN << "Data recovered for "
                << time_now - timeout_resume_start_time_ << "s";
        }
      } else {
        timeout_pause_start_time_ = 0.0;
        timeout_resume_start_time_ = 0.0;
      }
    }
  } else {
    is_timeout_paused_ = false;
    DebugDatas::Instance()->emplace("mission_status", 0);
  }
  DebugDatas::Instance()->emplace("mission_is_timeout_paused",
                                  is_timeout_paused_);
  DebugDatas::Instance()->emplace("mission_input_timeout",
                                  necessary_input_timeout);
  DebugDatas::Instance()->emplace("odom_x", odometry_.pose.position.x());
  DebugDatas::Instance()->emplace("odom_y", odometry_.pose.position.y());
  DebugDatas::Instance()->emplace("odom_theta", odometry_.pose.euler().z());
  DebugDatas::Instance()->emplace("odom_vx", odometry_.twist.linear.x());
  DebugDatas::Instance()->emplace("odom_vy", odometry_.twist.linear.y());
  DebugDatas::Instance()->emplace("odom_omega", odometry_.twist.angular.z());
  return true;
}

bool RosBridge::Publish() {
  auto mission_entry = planning_entry_->mission_entry_ptr();
  // Fault edge events (raised/cleared) are independent of the mission state
  diagnostic_bridge_->Publish();
  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle) {
    auto current_mission = mission_entry->GetCurrentMissionPtr();
    auto cmd = current_mission->GetTwist();

    Eigen::Vector3d cmd_vel(cmd.twist.linear.x(), cmd.twist.linear.y(),
                            cmd.twist.angular.z());
    Eigen::Vector3d actual_vel(odometry_.twist.linear.x(),
                               odometry_.twist.linear.y(),
                               odometry_.twist.angular.z());
    DebugDatas::Instance()->emplace("cmd_vx", cmd.twist.linear.x());
    DebugDatas::Instance()->emplace("cmd_vy", cmd.twist.linear.y());
    DebugDatas::Instance()->emplace("cmd_omega", cmd.twist.angular.z());
    cmd_vel =
        VelocityCalibration::Instance()->ComputeCommand(cmd_vel, actual_vel);
    cmd.twist.linear.x() = cmd_vel.x();
    cmd.twist.linear.y() = cmd_vel.y();
    cmd.twist.angular.z() = cmd_vel.z();

    const auto *integral_state =
        VelocityCalibration::Instance()->IntegralState();
    DebugDatas::Instance()->emplace("cmd_re_vx", cmd.twist.linear.x());
    DebugDatas::Instance()->emplace("cmd_re_vy", cmd.twist.linear.y());
    DebugDatas::Instance()->emplace("cmd_re_omega", cmd.twist.angular.z());

    geometry_msgs::msg::Twist cmd_msg;
    cmd_msg.linear.x = cmd.twist.linear.x();
    cmd_msg.linear.y = cmd.twist.linear.y();
    cmd_msg.linear.z = cmd.twist.linear.z();
    cmd_msg.angular.x = cmd.twist.angular.x();
    cmd_msg.angular.y = cmd.twist.angular.y();
    cmd_msg.angular.z = cmd.twist.angular.z();
    pub_cmd_->publish(cmd_msg);

    if (mission_entry->GetCurrentMissionType() == MissionConf::Recharging) {
      auto recharging = dynamic_cast<Recharging *>(
          mission_entry->GetMissionPtr(MissionConf::Recharging));
      uint8_t motion = 0;
      std::vector<float> para;
      if (recharging->TakePendingMotion(motion, para)) {
        mc_interface::msg::Motion motion_msg;
        motion_msg.header.stamp = node_->now();
        motion_msg.uuid = "planning";
        motion_msg.mode = mc_interface::msg::Mode::MODE_WBMC;
        motion_msg.motion = motion;
        motion_msg.para = para;
        pub_motion_->publish(motion_msg);
      } else {
        const auto inner_state = recharging->recharging_status();
        const bool is_walking_state =
            inner_state == RechargingFSMBase::StateType::searching ||
            inner_state == RechargingFSMBase::StateType::aligning ||
            inner_state == RechargingFSMBase::StateType::inserting ||
            inner_state == RechargingFSMBase::StateType::backing_off;
        if (is_walking_state) {
          mc_interface::msg::Motion motion_msg;
          motion_msg.header.stamp = node_->now();
          motion_msg.uuid = "planning";
          motion_msg.mode = mc_interface::msg::Mode::MODE_WBMC;
          motion_msg.motion = mc_interface::msg::MotionSkill::MOTION_GAIT_WALK;
          motion_msg.para = {static_cast<float>(cmd.twist.linear.x()),
                             static_cast<float>(cmd.twist.linear.y()),
                             static_cast<float>(cmd.twist.angular.z()),
                             1.0f}; // 1.0 = 精细化控制
          pub_motion_->publish(motion_msg);
        }
      }
    } else if (current_mission->mission_status() !=
               MissionFSMBase::StateType::standstill) {
      mc_interface::msg::Motion motion_msg;
      motion_msg.header.stamp = node_->now();
      motion_msg.uuid = "planning";
      motion_msg.mode = mc_interface::msg::Mode::MODE_WBMC;
      if (ConfigHelper::Instance()
              ->GetConfig()
              .robot_param()
              .robot_id()
              .appellation() == "d20") {
        motion_msg.motion = mc_interface::msg::MotionSkill::MOTION_GAIT_WALK;
      } else {
        motion_msg.motion = mc_interface::msg::MotionSkill::MOTION_GAIT_RUN;
      }
      motion_msg.para = {static_cast<float>(cmd.twist.linear.x()),
                         static_cast<float>(cmd.twist.linear.y()),
                         static_cast<float>(cmd.twist.angular.z())};
      pub_motion_->publish(motion_msg);
    }
  }
  tracking_bridge_->Publish();
  recharging_bridge_->Publish();
  following_bridge_->Publish();
  skill_bridge_->Publish();

  rclcpp::spin_some(node_);
  rate_.sleep();
  return true;
}

void RosBridge::sync_callback(
    const nav_msgs::msg::Odometry::SharedPtr &odom_msg,
    const nav_msgs::msg::OccupancyGrid::SharedPtr &occ_msg) {
  AINFO << "sync_callback, odom ts: " << odom_msg->header.stamp.sec << " "
        << odom_msg->header.stamp.nanosec
        << " occ ts: " << occ_msg->header.stamp.sec << " "
        << occ_msg->header.stamp.nanosec;
  odometry_.timestamp =
      odom_msg->header.stamp.sec + odom_msg->header.stamp.nanosec * 1e-9;
  odometry_.pose.position.x() = odom_msg->pose.pose.position.x;
  odometry_.pose.position.y() = odom_msg->pose.pose.position.y;
  odometry_.pose.position.z() = odom_msg->pose.pose.position.z;
  odometry_.pose.orientation.x() = odom_msg->pose.pose.orientation.x;
  odometry_.pose.orientation.y() = odom_msg->pose.pose.orientation.y;
  odometry_.pose.orientation.z() = odom_msg->pose.pose.orientation.z;
  odometry_.pose.orientation.w() = odom_msg->pose.pose.orientation.w;
  odometry_.twist.linear.x() = odom_msg->twist.twist.linear.x;
  odometry_.twist.linear.y() = odom_msg->twist.twist.linear.y;
  odometry_.twist.linear.z() = odom_msg->twist.twist.linear.z;
  odometry_.twist.angular.x() = odom_msg->twist.twist.angular.x;
  odometry_.twist.angular.y() = odom_msg->twist.twist.angular.y;
  odometry_.twist.angular.z() = odom_msg->twist.twist.angular.z;
  TransformStamped tf(odometry_.timestamp, "map", "base_link",
                      odometry_.pose.position, odometry_.pose.orientation);
  TfBuffer::Instance()->add(tf);
  auto it = timeout_states_.find(config_.sub_odom().name());
  if (it != timeout_states_.end()) {
    it->second.last_time = TimeoutReference(odometry_.timestamp);
  }

  it = timeout_states_.find(config_.sub_occupancy_grid().name());
  if (it != timeout_states_.end()) {
    it->second.last_time = TimeoutReference(odometry_.timestamp);
  }
  grid_map::GridMapRosConverter::fromOccupancyGrid(*occ_msg, "obstacle",
                                                   grid_map_);
  auto &matrix = grid_map_.get("obstacle");
  matrix = matrix.array() * (-2.55) + 255;
}

void RosBridge::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  odometry_.timestamp =
      msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
  odometry_.pose.position.x() = msg->pose.pose.position.x;
  odometry_.pose.position.y() = msg->pose.pose.position.y;
  odometry_.pose.position.z() = msg->pose.pose.position.z;
  odometry_.pose.orientation.x() = msg->pose.pose.orientation.x;
  odometry_.pose.orientation.y() = msg->pose.pose.orientation.y;
  odometry_.pose.orientation.z() = msg->pose.pose.orientation.z;
  odometry_.pose.orientation.w() = msg->pose.pose.orientation.w;
  odometry_.twist.linear.x() = msg->twist.twist.linear.x;
  odometry_.twist.linear.y() = msg->twist.twist.linear.y;
  odometry_.twist.linear.z() = msg->twist.twist.linear.z;
  odometry_.twist.angular.x() = msg->twist.twist.angular.x;
  odometry_.twist.angular.y() = msg->twist.twist.angular.y;
  odometry_.twist.angular.z() = msg->twist.twist.angular.z;
  TransformStamped tf(odometry_.timestamp, "map", "base_link",
                      odometry_.pose.position, odometry_.pose.orientation);
  TfBuffer::Instance()->add(tf);
  auto it = timeout_states_.find(config_.sub_odom().name());
  if (it != timeout_states_.end()) {
    it->second.last_time = TimeoutReference(odometry_.timestamp);
  }
}

void RosBridge::joint_state_callback(
    const sensor_msgs::msg::JointState::SharedPtr msg) {
  // using tf
  auto collision_checker =
      planning_entry_->preprocess_entry_ptr()->collision_checker();
  if (!collision_checker) {
    return;
  }
  std::unordered_map<std::string, Eigen::Isometry3d> link_transforms;
  const std::string target_frame = "base_link";
  for (const auto &link_name : collision_checker->getLinkNames()) {
    try {
      geometry_msgs::msg::TransformStamped transform_stamped =
          tf_buffer_->lookupTransform(target_frame, link_name,
                                      tf2::TimePointZero);

      Eigen::Isometry3d transform =
          tf2::transformToEigen(transform_stamped.transform);
      link_transforms[link_name] = transform;
      ADEBUG << link_name << ": "
             << transform.rotation().eulerAngles(0, 1, 2).transpose() << " "
             << transform.translation().transpose();
    } catch (const tf2::TransformException &ex) {
      AWARN << "Could not transform " << target_frame << " to " << link_name
            << ": " << ex.what();
      continue;
    }
  }
  collision_checker->updateTransforms(link_transforms);
  auto it = timeout_states_.find(config_.sub_joint_state().name());
  if (it != timeout_states_.end()) {
    it->second.last_time = TimeoutReference(odometry_.timestamp);
  }
}

void RosBridge::local_octomap_callback(
    const octomap_msgs::msg::Octomap::SharedPtr msg) {
  auto octree_ptr = std::shared_ptr<const octomap::OcTree>(
      dynamic_cast<octomap::OcTree *>(octomap_msgs::msgToMap(*msg)),
      [](const octomap::OcTree *ptr) { delete ptr; });

  if (!octree_ptr) {
    AERROR << "Failed to convert octomap message to OcTree";
    return;
  }
  auto it = timeout_states_.find(config_.sub_octomap().name());
  if (it != timeout_states_.end()) {
    it->second.last_time = TimeoutReference(odometry_.timestamp);
  }
  auto collision_checker =
      planning_entry_->preprocess_entry_ptr()->collision_checker();
  collision_checker->updateOctomap(octree_ptr);
}

void RosBridge::occupancy_callback(
    const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
  auto it = timeout_states_.find(config_.sub_occupancy_grid().name());
  if (it != timeout_states_.end()) {
    it->second.last_time = TimeoutReference(
        msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9);
  }
  grid_map::GridMapRosConverter::fromOccupancyGrid(*msg, "obstacle", grid_map_);
  auto &matrix = grid_map_.get("obstacle");
  matrix = matrix.array() * (-2.55) + 255;
}

void RosBridge::global_occupancy_callback(
    const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
  if (!config_.enable_global_map_reload()) {
    return;
  }
  static uint32_t last_load_sec = 0;
  if (msg->header.stamp.sec == last_load_sec)
    return;
  last_load_sec = msg->header.stamp.sec;
  AINFO << "GlobalOccupancyGrid update.";

  grid_map::GridMap map;
  grid_map::GridMapRosConverter::fromOccupancyGrid(*msg, "obstacle", map);
  auto &matrix = map.get("obstacle");
  matrix =
      (matrix.array() < 0 || matrix.array() > 30)
          .select(
              Eigen::MatrixXf::Constant(matrix.rows(), matrix.cols(), 0.0f),
              Eigen::MatrixXf::Constant(matrix.rows(), matrix.cols(), 255.0f));

  auto global_map_manager =
      planning_entry_->preprocess_entry_ptr()->global_map_manager();
  if (global_map_manager) {
    global_map_manager->UpdateMap(map);
  }
}

void RosBridge::map_switch_callback(
    const std_msgs::msg::String::SharedPtr msg) {
  AINFO << "Map switch: " << msg->data;
  auto global_map_manager =
      planning_entry_->preprocess_entry_ptr()->global_map_manager();
  if (global_map_manager) {
    global_map_manager->ReloadMap(msg->data);
  }
}
void RosBridge::map_update_callback(
    const common_interface::msg::PathFile::SharedPtr msg) {
  AINFO << "Map update: " << msg->path;
  auto global_map_manager =
      planning_entry_->preprocess_entry_ptr()->global_map_manager();
  if (global_map_manager) {
    global_map_manager->ReloadMap(msg->path);
  }
}

bool RosBridge::checkTimeout() {
  double now = TimeKeeper::TimeNowSec();
  bool any_timeout = false;
  DebugDatas::Instance()->emplace("planning_ts", now);
  for (auto &[name, state] : timeout_states_) {
    DebugDatas::Instance()->emplace(name, state.last_time);
    if (state.is_active && !state.is_active()) {
      state.timed_out = false;
      continue;
    }
    if ((now - state.last_time) > state.threshold) {
      AERROR << name << " timeout! Last received: " << (now - state.last_time)
             << "s ago, threshold: " << state.threshold << "s";
      state.timed_out = true;
      any_timeout = true;
    } else {
      state.timed_out = false;
    }
  }
  return any_timeout;
}

void RosBridge::OnInitPose(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
  odometry_.timestamp =
      msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
  odometry_.pose.position.x() = msg->pose.pose.position.x;
  odometry_.pose.position.y() = msg->pose.pose.position.y;
  odometry_.pose.position.z() = msg->pose.pose.position.z;
  odometry_.pose.orientation.x() = msg->pose.pose.orientation.x;
  odometry_.pose.orientation.y() = msg->pose.pose.orientation.y;
  odometry_.pose.orientation.z() = msg->pose.pose.orientation.z;
  odometry_.pose.orientation.w() = msg->pose.pose.orientation.w;
  odometry_.twist.linear.x() = 0;
  odometry_.twist.linear.y() = 0;
  odometry_.twist.linear.z() = 0;
  odometry_.twist.angular.x() = 0;
  odometry_.twist.angular.y() = 0;
  odometry_.twist.angular.z() = 0;
  TransformStamped tf(odometry_.timestamp, "map", "base_link",
                      odometry_.pose.position, odometry_.pose.orientation);
  TfBuffer::Instance()->add(tf);
}

void RosBridge::OnGoalPose(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  auto mission_entry = planning_entry_->mission_entry_ptr();
  auto mission_is_off = [&mission_entry]() {
    return mission_entry->GetCurrentMissionType() == MissionConf::Idle ||
           mission_entry->GetCurrentMissionPtr()->mission_status() ==
               MissionFSMBase::StateType::off;
  };
  if (!mission_is_off()) {
    AINFO << "Mission is not off, stop current mission.";
    mission_entry->GetCurrentMissionPtr()->stop();
  } else if (mission_entry->GetCurrentMissionType() == MissionConf::Idle) {
    AINFO << "Set mission to Tracking.";
    mission_entry->SetCurrentMissionType(MissionConf::Tracking);
    AINFO << "Start Tracking mission.";
    auto tracking = dynamic_cast<Tracking *>(
        mission_entry->GetMissionPtr(MissionConf::Tracking));

    Pose3d target_pose;
    target_pose.position.x() = msg->pose.position.x;
    target_pose.position.y() = msg->pose.position.y;
    target_pose.position.z() = msg->pose.position.z;
    target_pose.orientation.x() = msg->pose.orientation.x;
    target_pose.orientation.y() = msg->pose.orientation.y;
    target_pose.orientation.z() = msg->pose.orientation.z;
    target_pose.orientation.w() = msg->pose.orientation.w;

    const std::string &frame_id = msg->header.frame_id;
    if (!frame_id.empty() && frame_id != "map") {
      Transform3d tf;
      if (TfBuffer::Instance()->get("map", frame_id, tf)) {
        Eigen::Vector3d pos(target_pose.position.x(), target_pose.position.y(),
                            target_pose.position.z());
        Eigen::Quaterniond ori = target_pose.orientation;
        Eigen::Vector3d transformed_pos = tf * pos;
        Eigen::Quaterniond transformed_ori =
            Eigen::Quaterniond(tf.rotation()) * ori;
        target_pose.position.x() = transformed_pos.x();
        target_pose.position.y() = transformed_pos.y();
        target_pose.position.z() = transformed_pos.z();
        target_pose.orientation = transformed_ori;
        AINFO << "Transformed goal from " << frame_id << " to map: ("
              << transformed_pos.x() << ", " << transformed_pos.y() << ")";
      } else {
        AWARN << "Failed to get transform from " << frame_id
              << " to map, using original pose";
      }
    }

    tracking->setTargetPose(target_pose, true);
    mission_entry->GetCurrentMissionPtr()->start();
  }
}

} // namespace communication
} // namespace planning
