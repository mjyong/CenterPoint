//
// Created by Kilo Code on 2026-02-28.
// 模式切换节点：根据服务信号切换建图与定位模式
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sched.h>
#include <string>
#include <thread>
#include <opencv2/opencv.hpp>

#include "core/system/loc_system.h"
#include "core/system/slam.h"
#include "io/yaml_io.h"
#include "nav2_interface/action/mapping.hpp"
#include "nav2_interface/action/relocation.hpp"
#include "nav2_interface/msg/nav_state.hpp"
#include "std_msgs/msg/string.hpp"

using namespace lightning;

DEFINE_string(config, "./config/default.yaml", "配置文件");

class ModeSwitchNode : public rclcpp::Node {
public:
    ModeSwitchNode() : Node("mode_switch_node"), running_(true) {
        // 获取配置文件路径（优先使用环境变量，否则使用命令行参数）
        const char* config_env = std::getenv("config_path");
        if (config_env != nullptr) {
            config_path_ = config_env;
        } else {
            config_path_ = FLAGS_config;
        }

        // 创建 Mapping action server
        mapping_action_server_ = rclcpp_action::create_server<nav2_interface::action::Mapping>(
            this, "/nav/mapping",
            std::bind(&ModeSwitchNode::handleMappingGoal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&ModeSwitchNode::handleMappingCancel, this, std::placeholders::_1),
            std::bind(&ModeSwitchNode::handleMappingAccepted, this, std::placeholders::_1));

        // 创建 Relocation action server
        relocation_action_server_ = rclcpp_action::create_server<nav2_interface::action::Relocation>(
            this, "/nav/relocation",
            std::bind(&ModeSwitchNode::handleRelocationGoal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&ModeSwitchNode::handleRelocationCancel, this, std::placeholders::_1),
            std::bind(&ModeSwitchNode::handleRelocationAccepted, this, std::placeholders::_1));

        // 创建 /map/switch 订阅（QoS: reliable().transient_local().keep_last(1)）
        // 收到地图切换通知后切换地图并重启定位；重定位功能不再负责切换地图
        rclcpp::QoS map_switch_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
        map_switch_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/map/switch", map_switch_qos,
            std::bind(&ModeSwitchNode::onMapSwitch, this, std::placeholders::_1));

        // 启动后默认开启定位模式，读取配置文件开启定位
        startDefaultLocalization();

        LOG(INFO) << "ModeSwitchNode 初始化完成，等待 action 请求...";
    }

    ~ModeSwitchNode() {
        running_ = false;  // 通知所有工作线程退出

        // 按依赖顺序回收线程：
        // 1. 先 join 建图线程（其 cleanup 可能启动新定位，需先完成）
        {
            std::lock_guard<std::mutex> lock(mode_mutex_);
            if (slam_system_) {
                slam_system_->CancelSpin();
            }
        }
        if (slam_spin_thread_.joinable()) {
            slam_spin_thread_.join();
        }
        if (slam_thread_.joinable()) {
            slam_thread_.join();
        }
        // 2. join 重定位线程
        if (relocation_thread_.joinable()) {
            relocation_thread_.join();
        }
        // 3. 最后停止并回收定位模块
        stopLocalization();

        LOG(INFO) << "ModeSwitchNode 已销毁";
    }

private:
    enum class Mode { MAPPING, LOCALIZATION, IDLE };

    std::atomic<Mode> current_mode_{Mode::LOCALIZATION};
    std::string config_path_;

    std::unique_ptr<SlamSystem> slam_system_;
    std::shared_ptr<LocSystem> loc_system_;  // shared_ptr 让 locLoop 持有引用，避免 UAF
    std::thread slam_thread_;
    std::thread slam_spin_thread_;   // SLAM executor spin 线程，持续处理 IMU/LiDAR 回调
    std::thread localization_thread_;   // 持久定位后台线程
    std::thread relocation_thread_;     // 重定位 action 执行线程
    std::atomic<bool> running_{true};
    std::atomic<bool> slam_running_{false};
    std::atomic<bool> localization_running_{false};

    std::mutex mode_mutex_;  // 改为普通互斥锁，避免递归锁的滥用

    // Mapping 相关
    std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_interface::action::Mapping>> active_mapping_goal_handle_;
    std::string current_map_uuid_;  // 当前已加载的地图 UUID（文件夹名）
    rclcpp_action::Server<nav2_interface::action::Mapping>::SharedPtr mapping_action_server_;

    // Relocation 相关
    std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_interface::action::Relocation>> active_relocation_goal_handle_;
    std::shared_ptr<const nav2_interface::action::Relocation::Goal> current_relocation_goal_;
    rclcpp_action::Server<nav2_interface::action::Relocation>::SharedPtr relocation_action_server_;
    nav2_interface::action::Mapping::Feedback::SharedPtr mapping_feedback;
    std::mutex feedback_mutex_;  // 保护 mapping_feedback 的访问

    // 地图切换 (/map/switch) 相关
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr map_switch_sub_;
    // 当前已加载地图的完整路径（来自 /map/switch 指令的 path，或建图后拼出的 "data/<uuid>"）
    // 重定位初始化时直接使用此路径，而非用 current_map_uuid_ 重新拼接
    std::string current_map_path_;
    // 建图结束时保存的最后一帧关键帧位姿（优化后），作为后续重启定位的初始位姿来源
    SE3 last_loc_pose_;
    bool has_last_loc_pose_{false};
    std::mutex last_loc_pose_mutex_;

    // ==================== Mapping Action 处理 ====================
    rclcpp_action::GoalResponse handleMappingGoal(
        const rclcpp_action::GoalUUID& uuid,
        std::shared_ptr<const nav2_interface::action::Mapping::Goal> goal) {
        (void)uuid;
        (void)goal;
        std::lock_guard<std::mutex> lock(mode_mutex_);
        if (active_mapping_goal_handle_) {
            LOG(WARNING) << "已有活动的 Mapping goal，拒绝新 goal";
            return rclcpp_action::GoalResponse::REJECT;
        }
        // 重定位进行中时拒绝建图请求，避免建图停止定位导致重定位监控循环异常
        if (active_relocation_goal_handle_) {
            LOG(WARNING) << "正在重定位，拒绝建图请求";
            return rclcpp_action::GoalResponse::REJECT;
        }
        // 地图切换进行中时拒绝建图请求
        if (current_mode_ == Mode::IDLE) {
            LOG(WARNING) << "系统正在切换地图，拒绝建图请求";
            return rclcpp_action::GoalResponse::REJECT;
        }
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handleMappingCancel(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_interface::action::Mapping>>& goal_handle) {
        std::lock_guard<std::mutex> lock(mode_mutex_);
        if (active_mapping_goal_handle_ && active_mapping_goal_handle_->get_goal_id() == goal_handle->get_goal_id()) {
            LOG(INFO) << "收到 Mapping goal 取消请求";
            return rclcpp_action::CancelResponse::ACCEPT;
        }
        return rclcpp_action::CancelResponse::REJECT;
    }

    void handleMappingAccepted(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_interface::action::Mapping>>& goal_handle) {
        // 提前注册 goal handle，避免 onMapSwitch 在 executeMapping 设置前抢入
        {
            std::lock_guard<std::mutex> lock(mode_mutex_);
            active_mapping_goal_handle_ = goal_handle;
        }
        // 等待之前的线程结束（如果有）
        if (slam_thread_.joinable()) {
            LOG(INFO) << "等待之前的 SLAM 线程结束...";
            slam_thread_.join();
        }
        // 启动新线程执行建图
        slam_thread_ = std::thread(&ModeSwitchNode::executeMapping, this, goal_handle);
    }

    void executeMapping(const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_interface::action::Mapping>> goal_handle) {
        LOG(INFO) << "执行 Mapping action";
        rclcpp::Rate loop_rate(500);  // 500 Hz
        const auto goal = goal_handle->get_goal();
        {
            std::lock_guard<std::mutex> lock(feedback_mutex_);
            mapping_feedback = std::make_shared<nav2_interface::action::Mapping::Feedback>();
        }
        auto result = std::make_shared<nav2_interface::action::Mapping::Result>();

        // goal handle 已在 handleMappingAccepted 中提前注册，此处仅设置 map_uuid
        {
            std::lock_guard<std::mutex> lock(mode_mutex_);
            // 若 map_uuid 为空，则随机生成 UUID
            if (goal->map_uuid.empty()) {
                current_map_uuid_ = generateUUID();
                LOG(INFO) << "map_uuid 为空，随机生成 UUID: " << current_map_uuid_;
            } else {
                current_map_uuid_ = goal->map_uuid;
            }
        }

        // RAII 清理：无论正常结束还是异常，都确保资源被释放
        // 标记是否已由 stopSlamAndSaveMap 启动了定位，避免 cleanup 重复启动
        bool loc_already_started = false;
        auto cleanup = [this, &loc_already_started]() {
            {
                std::lock_guard<std::mutex> lock(mode_mutex_);
                slam_running_ = false;
                if (slam_system_) {
                    slam_system_->CancelSpin();  // 让阻塞的 Spin() 返回
                }
            }
            // 等待 spin 线程结束
            if (slam_spin_thread_.joinable()) {
                slam_spin_thread_.join();
            }
            {
                std::lock_guard<std::mutex> lock(mode_mutex_);
                slam_system_.reset();
                active_mapping_goal_handle_.reset();
            }
            // 建图结束后恢复定位功能（节点未退出时）
            // 若 stopSlamAndSaveMap 已启动定位，则不再重复启动
            if (running_ && rclcpp::ok() && !loc_already_started) {
                LOG(INFO) << "建图结束，恢复定位系统";
                startDefaultLocalization();
            }
        };
        std::shared_ptr<void> guard(nullptr, [&](void*) { cleanup(); });

        LOG(INFO) << "开始 Mapping，地图 UUID: " << current_map_uuid_;

        // 收到建图请求：关闭定位功能，进入建图，避免定位线程与建图线程同时占用传感器数据
        // 先设置模式为 MAPPING，再停止定位，保证状态原子性（避免其他线程在此窗口内发起重定位）
        {
            std::lock_guard<std::mutex> lock(mode_mutex_);
            current_mode_ = Mode::MAPPING;
        }
        LOG(INFO) << "建图请求：停止定位系统";
        stopLocalization();

        try {
            SlamSystem::Options opts;
            opts.online_mode_ = true;
            slam_system_ = std::make_unique<SlamSystem>(opts);
            if (!slam_system_->Init(config_path_)) {
                LOG(ERROR) << "SlamSystem 初始化失败";
                result->message = "SlamSystem initialization failed";
                goal_handle->abort(result);
                return;
            }
            slam_system_->StartSLAM("dynamic_map");

            // 启动地图监控，当位移超过3米或旋转超过30度时发送地图反馈
            auto weak_goal_handle = std::weak_ptr<rclcpp_action::ServerGoalHandle<nav2_interface::action::Mapping>>(goal_handle);
            std::string map_uuid = current_map_uuid_; // 捕获副本
            slam_system_->StartMapMonitor(5.0, 90.0, [weak_goal_handle, map_uuid, this](const std::vector<uint8_t>& data, int width, int height, float resolution, float origin_x, float origin_y) {
                auto goal_handle = weak_goal_handle.lock();
                if (!goal_handle || !goal_handle->is_active()) {
                    return;
                }
                // 检查目标是否正在取消
                if (goal_handle->is_canceling()) {
                    LOG(INFO) << "地图监控回调：目标正在取消，跳过反馈";
                    return;
                }
                // 检查SLAM是否仍在运行
                if (!slam_running_) {
                    LOG(INFO) << "地图监控回调：SLAM已停止，跳过反馈";
                    return;
                }
                // 将灰度数据编码为 PNG 格式的二进制数据
                cv::Mat gray(height, width, CV_8UC1, const_cast<uint8_t*>(data.data()));
                std::vector<uint8_t> png_buffer;
                std::vector<int> compression_params = {cv::IMWRITE_PNG_COMPRESSION, 3};
                bool encode_ok = false;
                try {
                    encode_ok = cv::imencode(".png", gray, png_buffer, compression_params);
                } catch (const cv::Exception& e) {
                    LOG(ERROR) << "图像编码失败: " << e.what();
                    return;
                } catch (const std::exception& e) {
                    LOG(ERROR) << "图像编码异常: " << e.what();
                    return;
                }
                if (!encode_ok || png_buffer.empty()) {
                    LOG(ERROR) << "PNG 编码失败，png_buffer 为空";
                    return;
                }
                // 发布反馈
                {
                    std::lock_guard<std::mutex> lock(feedback_mutex_);
                    mapping_feedback->state = nav2_interface::msg::NavState::NAV_STA_MAP_DOING;
                    mapping_feedback->map_uuid = map_uuid;
                    mapping_feedback->map2d_name = "map.png";
                    geometry_msgs::msg::Pose origin;
                    origin.position.x = origin_x;
                    origin.position.y = origin_y;
                    origin.position.z = 0.0;
                    mapping_feedback->origin = origin;
                    mapping_feedback->resolution = resolution;
                    mapping_feedback->map = png_buffer;
                }
                try {
                    goal_handle->publish_feedback(mapping_feedback);
                    LOG(INFO) << "地图监控触发，发送地图反馈，大小: " << png_buffer.size() << " 字节，分辨率："<<resolution;
                } catch (const std::exception& e) {
                    LOG(ERROR) << "发布反馈失败: " << e.what();
                }
            });

            {
                std::lock_guard<std::mutex> lock(mode_mutex_);
                slam_running_ = true;
            }

            // 在独立线程中持续 spin executor，确保 IMU/LiDAR 回调及时处理
            // 使用 Spin() 持续阻塞模式，与直接 launch 行为一致
            slam_spin_thread_ = std::thread([this]() {
                if (slam_system_) {
                    slam_system_->Spin();
                }
            });

            auto last_feedback_time = this->now();
            // 主循环：仅检查取消状态，不再驱动 executor
            while (rclcpp::ok() && running_ && slam_running_) {
                if (goal_handle->is_canceling()) {
                    LOG(INFO) << "Mapping goal 被取消，停止建图并保存地图";
                    stopSlamAndSaveMap(current_map_uuid_, true);
                    loc_already_started = true;
                    return;  // stopSlamAndSaveMap 已发送结果并启动定位，cleanup 仅做剩余清理
                }

                // 发布反馈
                // auto now = this->now();
                // if ((now - last_feedback_time).seconds() >= 5.0) {
                //     std::lock_guard<std::mutex> lock(feedback_mutex_);
                //     goal_handle->publish_feedback(mapping_feedback);
                //     last_feedback_time = now;
                // }

                loop_rate.sleep();
            }

            // 正常退出（非取消）
            if (!slam_running_) {
                LOG(INFO) << "SLAM 被外部停止，保存地图";
                stopSlamAndSaveMap(current_map_uuid_, false);
                loc_already_started = true;
            } else if (!rclcpp::ok() || !running_) {
                LOG(INFO) << "ROS2 关闭或节点停止，保存地图";
                stopSlamAndSaveMap(current_map_uuid_, false);
                loc_already_started = true;
            }
        } catch (const std::exception& e) {
            LOG(ERROR) << "Mapping 执行异常: " << e.what();
            result->message = std::string("Exception: ") + e.what();
            goal_handle->abort(result);
        }
    }

    void stopSlamAndSaveMap(const std::string& map_uuid, bool is_cancel) {
        // 先取出需要清理的资源，避免在锁内做耗时操作或二次加锁
        std::unique_ptr<SlamSystem> slam_system_to_stop;
        std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_interface::action::Mapping>> goal_handle_to_complete;
        {
            std::lock_guard<std::mutex> lock(mode_mutex_);
            if (!slam_running_) {
                return;  // 已经停止
            }
            slam_running_ = false;
            // 停止地图监控
            if (slam_system_) {
                slam_system_->StopMapMonitor();
                slam_system_->CancelSpin();  // 让阻塞的 Spin() 返回
            }
            slam_system_to_stop = std::move(slam_system_);
            goal_handle_to_complete = std::exchange(active_mapping_goal_handle_, nullptr);
        }

        // 等待 spin 线程结束（CancelSpin 已让 Spin() 返回）
        if (slam_spin_thread_.joinable()) {
            slam_spin_thread_.join();
        }

        LOG(INFO) << "停止 SLAM 并保存地图到 data/" << map_uuid;
        SE3 last_keyframe_pose;  // 保存地图（含全局优化）后获取优化后的关键帧位姿
        if (slam_system_to_stop) {
            //slam_system_to_stop->SaveMap("data/" + map_uuid);
            slam_system_to_stop->SaveMapByTemplate("data/" + map_uuid);
            // SaveMap 内部执行全局优化，优化后关键帧位姿才更新，故在此处获取
            last_keyframe_pose = slam_system_to_stop->GetLastKeyframePose();
        }
        // 释放 SLAM 资源
        slam_system_to_stop.reset();

        // 检查地图数据是否已实际保存（SaveMap 在空关键帧时会提前返回）
        std::string map_data_path = "data/" + map_uuid;
        if (!std::filesystem::exists(map_data_path + "/global.pcd") &&
            !std::filesystem::exists(map_data_path + "/3d/global.pcd")) {
            LOG(WARNING) << "No map data saved (empty map), falling back to default localization";
            startDefaultLocalization();
            if (goal_handle_to_complete && goal_handle_to_complete->is_active()) {
                auto result = std::make_shared<nav2_interface::action::Mapping::Result>();
                result->message = is_cancel ? "Mapping cancelled, map empty" : "Mapping finished, map empty";
                goal_handle_to_complete->succeed(result);
            }
            return;
        }

        // 建图结束后将新地图路径写入配置文件，后续启动自动加载
        updateConfigMapPath(map_data_path);

        // 获取地图的绝对路径
        std::string abs_map_path = std::filesystem::absolute(map_data_path).string();

        LOG(INFO) << "SLAM 已停止，地图已保存，绝对路径: " << abs_map_path;

        // 保存建图最后一帧关键帧位姿，供后续地图切换重启定位时作为初始位姿
        {
            std::lock_guard<std::mutex> lock(last_loc_pose_mutex_);
            last_loc_pose_ = last_keyframe_pose;
            has_last_loc_pose_ = true;
        }

        // 更新当前已加载地图 UUID 和路径
        {
            std::lock_guard<std::mutex> lock(mode_mutex_);
            current_map_uuid_ = map_uuid;
            current_map_path_ = map_data_path;
        }

        // 先启动定位，确保系统就绪后再通知客户端建图结果
        LOG(INFO) << "建图结束，启动定位，地图: " << map_data_path
                  << "，初始位姿: " << last_keyframe_pose.translation().transpose();
        bool loc_ok = startLocalizationWithMap(map_data_path, last_keyframe_pose);

        // 定位启动失败时尝试回退到默认地图
        if (!loc_ok) {
            LOG(ERROR) << "建图后启动定位失败，尝试加载默认地图";
            startDefaultLocalization();
        }

        // 根据是否取消，使用 canceled 或 succeed 通知客户端
        if (goal_handle_to_complete && goal_handle_to_complete->is_active()) {
            auto result = std::make_shared<nav2_interface::action::Mapping::Result>();
            if (is_cancel) {
                result->message = "Mapping cancelled, map saved";
                result->map_path = abs_map_path;
                goal_handle_to_complete->canceled(result);
            } else {
                result->message = "Mapping finished, map saved";
                result->map_path = abs_map_path;
                goal_handle_to_complete->succeed(result);
            }
        }
    }

    // ==================== Relocation Action 处理 ====================
    rclcpp_action::GoalResponse handleRelocationGoal(
        const rclcpp_action::GoalUUID& uuid,
        std::shared_ptr<const nav2_interface::action::Relocation::Goal> goal) {
        (void)uuid;
        std::lock_guard<std::mutex> lock(mode_mutex_);
        if (active_relocation_goal_handle_) {
            LOG(WARNING) << "已有活动的 Relocation goal，拒绝新 goal";
            return rclcpp_action::GoalResponse::REJECT;
        }
        // 正在建图时拒绝重定位
        if (active_mapping_goal_handle_ || slam_running_) {
            LOG(WARNING) << "正在建图，拒绝重定位请求";
            return rclcpp_action::GoalResponse::REJECT;
        }
        // 地图切换进行中时拒绝重定位，避免访问正在销毁/重建的 loc_system_
        if (current_mode_ == Mode::IDLE) {
            LOG(WARNING) << "系统正在切换地图，拒绝重定位请求";
            return rclcpp_action::GoalResponse::REJECT;
        }
        // 重定位不负责切换地图：若请求的 map_uuid 与当前已加载地图不同，直接拒绝
        if (goal->map_uuid != current_map_uuid_) {
            LOG(WARNING) << "重定位请求的 map_uuid(" << goal->map_uuid
                         << ") 与当前已加载地图(" << current_map_uuid_
                         << ") 不同，拒绝。请先通过 /map/switch 切换地图";
            return rclcpp_action::GoalResponse::REJECT;
        }
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handleRelocationCancel(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_interface::action::Relocation>>& goal_handle) {
        std::lock_guard<std::mutex> lock(mode_mutex_);
        if (active_relocation_goal_handle_ && active_relocation_goal_handle_->get_goal_id() == goal_handle->get_goal_id()) {
            LOG(INFO) << "收到 Relocation goal 取消请求";
            return rclcpp_action::CancelResponse::ACCEPT;
        }
        return rclcpp_action::CancelResponse::REJECT;
    }

    void handleRelocationAccepted(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_interface::action::Relocation>>& goal_handle) {
        // 提前注册 goal handle，避免 onMapSwitch 在 executeRelocation 设置前抢入
        {
            std::lock_guard<std::mutex> lock(mode_mutex_);
            active_relocation_goal_handle_ = goal_handle;
        }
        // 等待之前的重定位线程结束
        if (relocation_thread_.joinable()) {
            LOG(INFO) << "等待之前的重定位线程结束...";
            relocation_thread_.join();
        }
        relocation_thread_ = std::thread(&ModeSwitchNode::executeRelocation, this, goal_handle);
    }

    // ==================== 默认定位模式（启动即开启） ====================
    // 启动后默认开启定位模式，读取配置文件开启定位
    void startDefaultLocalization() {
        std::lock_guard<std::mutex> lock(mode_mutex_);
        if (loc_system_) {
            LOG(WARNING) << "定位系统已在运行，跳过默认启动";
            return;
        }

        // 从配置文件读取 system.map_path，解析出当前已加载的地图 UUID 和路径
        try {
            YAML_IO yaml(config_path_);
            std::string map_path = yaml.GetValue<std::string>("system", "map_path");
            current_map_uuid_ = extractMapUuid(map_path);
            current_map_path_ = map_path;
            LOG(INFO) << "启动时从配置文件加载地图 UUID: " << current_map_uuid_
                      << "，路径: " << current_map_path_;
        } catch (const std::exception& e) {
            LOG(WARNING) << "读取 system.map_path 失败，current_map_uuid_ 为空: " << e.what();
            current_map_uuid_.clear();
            current_map_path_.clear();
        }

        try {
            LocSystem::Options opts;
            opts.pub_tf_ = true;
            loc_system_ = std::make_shared<LocSystem>(opts);
            if (!loc_system_->Init(config_path_)) {
                LOG(ERROR) << "默认 LocSystem 初始化失败";
                loc_system_.reset();
                return;
            }
            // 从 LocSystem 同步实际的地图 UUID 和路径（可能已回退到默认地图）
            current_map_uuid_ = loc_system_->GetCurrentMapUuid();
            if (loc_system_->IsUsingDefaultMap()) {
                current_map_path_.clear();
                LOG(WARNING) << "配置地图无效，已加载内置默认地图，UUID: " << current_map_uuid_;
            }
            localization_running_ = true;
            current_mode_ = Mode::LOCALIZATION;
            localization_thread_ = std::thread(&ModeSwitchNode::locLoop, this, loc_system_);
            LOG(INFO) << "默认定位模式已启动（配置文件: " << config_path_ << "）";
        } catch (const std::exception& e) {
            LOG(ERROR) << "启动默认定位异常: " << e.what();
            loc_system_.reset();
            localization_running_ = false;
        }
    }

    // 使用指定地图和初始位姿启动定位（建图结束后调用），返回是否成功
    bool startLocalizationWithMap(const std::string& map_path, const SE3& init_pose) {
        // 先确保之前的定位已停止
        stopLocalization();

        try {
            LocSystem::Options opts;
            opts.pub_tf_ = true;
            {
                std::lock_guard<std::mutex> lock(mode_mutex_);
                loc_system_ = std::make_shared<LocSystem>(opts);
            }
            if (!loc_system_->Init(config_path_, false, false)) {
                LOG(ERROR) << "LocSystem 初始化失败";
                loc_system_.reset();
                return false;
            }

            // 锁定初始位姿状态，防止等待 INS 期间 INS 回调抢占初始化
            loc_system_->LockInitPose();

            // 等待接收 INS 数据（最多 5 秒），SwitchMapAndReinit 内部会优先使用 INS 位姿
            LOG(INFO) << "等待 INS 数据...";
            for (int i = 0; i < 200 && rclcpp::ok() && running_; ++i) {
                loc_system_->SpinSome();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            bool success = loc_system_->SwitchMapAndReinit(map_path, init_pose);
            if (!success) {
                LOG(ERROR) << "地图切换和重新初始化失败";
                loc_system_.reset();
                return false;
            }

            // 启动定位后台线程
            {
                std::lock_guard<std::mutex> lock(mode_mutex_);
                localization_running_ = true;
                current_mode_ = Mode::LOCALIZATION;
            }
            if (localization_thread_.joinable()) {
                localization_thread_.join();
            }
            localization_thread_ = std::thread(&ModeSwitchNode::locLoop, this, loc_system_);
            LOG(INFO) << "定位已启动，地图: " << map_path;
            return true;
        } catch (const std::exception& e) {
            LOG(ERROR) << "启动定位异常: " << e.what();
            loc_system_.reset();
            return false;
        }
    }

    // 定位后台循环：接收 shared_ptr 确保生命周期安全
    // 即使 stopLocalization reset 了成员 loc_system_，本线程持有的 loc 副本仍有效
    void locLoop(std::shared_ptr<LocSystem> loc) {
        // 创建定时器检查初始化超时（10Hz）
        // loc 是 shared_ptr 副本，生命周期独立于 loc_system_ 成员
        auto check_timer = loc->CreateWallTimer(
            std::chrono::milliseconds(100),
            [loc]() {
                loc->CheckInitTimeout();
            });

        // Spin 阻塞运行，让 MultiThreadedExecutor 用线程池分发回调
        loc->Spin();
        LOG(INFO) << "定位后台线程退出";
    }

    // ==================== /map/switch 地图切换通知 ====================
    // 从 map_path 中提取最后一级目录名作为 UUID
    // 例: "./data/new_map/" -> "new_map", "data/abc-123" -> "abc-123"
    static std::string extractMapUuid(const std::string& map_path) {
        std::string p = map_path;
        // 去掉末尾的 '/'
        while (!p.empty() && p.back() == '/') {
            p.pop_back();
        }
        auto pos = p.find_last_of('/');
        if (pos == std::string::npos) {
            return p;
        }
        return p.substr(pos + 1);
    }

    // /map/switch 订阅回调：切换地图并重启定位
    // path 内容为 "data/<uuid>" 或 "<uuid>"，统一提取最后一级目录名
    void onMapSwitch(const std_msgs::msg::String::SharedPtr msg) {
        if (!msg || msg->data.empty()) {
            LOG(WARNING) << "收到空的 /map/switch 消息，忽略";
            return;
        }
        std::string new_uuid = extractMapUuid(msg->data);
        LOG(INFO) << "收到 /map/switch: " << msg->data << "，解析 UUID: " << new_uuid;

        // 原子完成检查 + 模式切换，避免与 relocation 的竞态
        {
            std::lock_guard<std::mutex> lock(mode_mutex_);
            if (active_mapping_goal_handle_ || slam_running_) {
                LOG(WARNING) << "正在建图，不响应地图切换请求 (请求 UUID: " << new_uuid << ")";
                return;
            }
            if (active_relocation_goal_handle_) {
                LOG(WARNING) << "正在重定位，不响应地图切换请求 (请求 UUID: " << new_uuid << ")";
                return;
            }
            if (new_uuid == current_map_uuid_) {
                LOG(INFO) << "地图切换通知的 UUID 与当前已加载地图相同(" << current_map_uuid_
                          << ")，忽略";
                return;
            }
            current_mode_ = Mode::IDLE;
        }

        // 检查目标地图是否存在有效数据（index.txt），不存在则忽略切换请求
        {
            std::string map_dir = msg->data;
            while (!map_dir.empty() && map_dir.back() == '/') map_dir.pop_back();
            if (!std::filesystem::exists(map_dir + "/index.txt") &&
                !std::filesystem::exists(map_dir + "/3d/index.txt")) {
                LOG(WARNING) << "地图切换：目标地图无有效数据(无 index.txt): " << msg->data
                             << "，忽略切换请求";
                {
                    std::lock_guard<std::mutex> lock(mode_mutex_);
                    current_mode_ = Mode::LOCALIZATION;
                }
                return;
            }
        }

        // 重启定位用新地图
        // 初始位姿来源：建图结束时保存的最后一帧关键帧位姿（last_loc_pose_）；
        // 若无（如启动后未建图过），则用配置文件 yaml 的初始位姿（传单位位姿，
        // SwitchMapAndReinit 内部会优先使用 INS 位姿或 yaml 位姿）
        SE3 init_pose;  // 默认构造为单位位姿
        bool has_init_pose = false;
        {
            std::lock_guard<std::mutex> lock(last_loc_pose_mutex_);
            if (has_last_loc_pose_) {
                init_pose = last_loc_pose_;
                has_init_pose = true;
            }
        }

        LOG(INFO) << "地图切换：停止当前定位，切换到地图 " << new_uuid
                  << (has_init_pose ? "，使用建图最后一帧位姿作为初始位姿"
                                    : "，使用配置文件初始位姿");

        // 停止当前定位
        stopLocalization();

        try {
            // 直接使用 /map/switch 指令带来的 path 作为地图路径
            const std::string& map_path = msg->data;
            LocSystem::Options opts;
            opts.pub_tf_ = true;
            {
                std::lock_guard<std::mutex> lock(mode_mutex_);
                loc_system_ = std::make_shared<LocSystem>(opts);
            }
            if (!loc_system_->Init(config_path_, false, false)) {
                LOG(ERROR) << "地图切换：LocSystem 初始化失败，地图: " << map_path;
                loc_system_.reset();
                {
                    std::lock_guard<std::mutex> lock(mode_mutex_);
                    current_mode_ = Mode::LOCALIZATION;
                }
                return;
            }

            // 锁定初始位姿状态，防止等待 INS 期间 INS 回调抢占初始化
            loc_system_->LockInitPose();

            // 等待接收 INS 数据
            LOG(INFO) << "等待 INS 数据...";
            for (int i = 0; i < 100 && rclcpp::ok() && running_; ++i) {
                loc_system_->SpinSome();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            bool success = false;
            if (has_init_pose) {
                success = loc_system_->SwitchMapAndReinit(map_path, init_pose);
            } else {
                // 无最后一个定位点，用配置文件/yaml 的初始位姿启动（传单位位姿，
                // SwitchMapAndReinit 内部会优先使用 INS 位姿或 yaml 位姿）
                success = loc_system_->SwitchMapAndReinit(map_path, SE3());
            }
            if (!success) {
                LOG(ERROR) << "地图切换失败，地图: " << map_path;
                loc_system_.reset();
                {
                    std::lock_guard<std::mutex> lock(mode_mutex_);
                    current_mode_ = Mode::LOCALIZATION;
                }
                return;
            }

            // 更新当前已加载地图 UUID 和路径（路径来自切换指令的原始 path）
            {
                std::lock_guard<std::mutex> lock(mode_mutex_);
                current_map_uuid_ = new_uuid;
                current_map_path_ = msg->data;
                localization_running_ = true;
                current_mode_ = Mode::LOCALIZATION;
            }
            if (localization_thread_.joinable()) {
                localization_thread_.join();
            }
            localization_thread_ = std::thread(&ModeSwitchNode::locLoop, this, loc_system_);

            // 将新地图路径写入配置文件
            updateConfigMapPath(msg->data);

            LOG(INFO) << "地图切换完成，新地图: " << new_uuid << "，定位已重启";
        } catch (const std::exception& e) {
            LOG(ERROR) << "地图切换异常: " << e.what();
            loc_system_.reset();
            {
                std::lock_guard<std::mutex> lock(mode_mutex_);
                current_mode_ = Mode::LOCALIZATION;
            }
        }
    }

    // 使用 Boost.UUID 生成全球唯一的 UUID v4
    std::string generateUUID() {
        auto uuid = boost::uuids::random_generator()();
        return boost::uuids::to_string(uuid);
    }

    // 将地图路径直接写入配置文件 system.map_path，后续启动时自动加载
    void updateConfigMapPath(const std::string& map_path) {
        try {
            YAML_IO yaml(config_path_);
            yaml.SetValue<std::string>("system", "map_path", map_path);
            yaml.Save();
            LOG(INFO) << "已将地图路径写入配置文件: system.map_path = " << map_path;
        } catch (const std::exception& e) {
            LOG(ERROR) << "更新配置文件失败: " << e.what();
        }
    }

    // 停止并回收定位模块（先通知线程退出，再 join，最后销毁 loc_system_）
    void stopLocalization() {
        LOG(INFO) << "stopLocalization: 开始停止定位模块";
        // 1. 通知定位线程退出
        localization_running_ = false;

        // 2. 在锁内取出 shared_ptr 副本并 CancelSpin，确保线程安全
        std::shared_ptr<LocSystem> loc_copy;
        {
            std::lock_guard<std::mutex> lock(mode_mutex_);
            loc_copy = loc_system_;  // 引用计数 +1，locLoop 不会 UAF
        }
        if (loc_copy) {
            loc_copy->CancelSpin();
        }

        // 3. 等待定位线程结束（确保不再访问 loc_system_）
        if (localization_thread_.joinable()) {
            LOG(INFO) << "stopLocalization: 等待定位线程结束...";
            localization_thread_.join();
            LOG(INFO) << "stopLocalization: 定位线程已结束";
        } else {
            LOG(INFO) << "stopLocalization: 定位线程未启动，无需 join";
        }

        // 4. 销毁定位系统（线程已退出，安全）
        {
            std::lock_guard<std::mutex> lock(mode_mutex_);
            if (loc_system_) {
                LOG(INFO) << "stopLocalization: 销毁 LocSystem 实例";
                loc_system_.reset();
            } else {
                LOG(INFO) << "stopLocalization: loc_system_ 已为空";
            }
        }
        loc_copy.reset();  // 释放本地引用
        LOG(INFO) << "定位模块已停止并回收";
    }

    // ==================== Relocation Action 处理 ====================
    void executeRelocation(const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_interface::action::Relocation>> goal_handle) {
        LOG(INFO) << "执行 Relocation action";
        const auto goal = goal_handle->get_goal();
        auto feedback = std::make_shared<nav2_interface::action::Relocation::Feedback>();
        auto result = std::make_shared<nav2_interface::action::Relocation::Result>();

        // goal handle 已在 handleRelocationAccepted 中提前注册，此处仅保存 goal
        {
            std::lock_guard<std::mutex> lock(mode_mutex_);
            current_relocation_goal_ = goal;
        }

        // RAII 清理：仅清理 goal handle，不销毁 loc_system_（定位模块需持续运行）
        auto cleanup = [this]() {
            std::lock_guard<std::mutex> lock(mode_mutex_);
            active_relocation_goal_handle_.reset();
        };
        std::shared_ptr<void> guard(nullptr, [&](void*) { cleanup(); });

        LOG(INFO) << "重定位请求 - 地图 UUID: " << goal->map_uuid
                  << "，初始位姿: (" << goal->pose.position.x << ", "
                  << goal->pose.position.y << ", "
                  << goal->pose.position.z << ")";

        // 1. 先回收注销定位模块（重定位只做定位初始化，不切换地图）
        LOG(INFO) << "重定位：停止当前定位模块以重新初始化";
        stopLocalization();

        // 2. 用当前已加载地图（current_map_path_，来自 /map/switch 指令）+ goal 位姿做定位初始化（不切换地图）
        try {
            std::string loaded_map_path;
            {
                std::lock_guard<std::mutex> lock(mode_mutex_);
                loaded_map_path = current_map_path_;
            }
            if (loaded_map_path.empty()) {
                LOG(ERROR) << "当前无已加载地图，无法重定位初始化";
                result->message = "No map loaded, please switch map first";
                goal_handle->abort(result);
                return;
            }

            const std::string& map_path = loaded_map_path;
            geometry_msgs::msg::Pose init_pose_msg = goal->pose;

            // 转换位姿
            Eigen::Quaterniond q(init_pose_msg.orientation.w,
                                 init_pose_msg.orientation.x,
                                 init_pose_msg.orientation.y,
                                 init_pose_msg.orientation.z);
            Eigen::Vector3d t(init_pose_msg.position.x,
                              init_pose_msg.position.y,
                              init_pose_msg.position.z);
            SE3 init_pose(q, t);

            LocSystem::Options opts;
            opts.pub_tf_ = true;
            {
                std::lock_guard<std::mutex> lock(mode_mutex_);
                loc_system_ = std::make_shared<LocSystem>(opts);
            }
            if (!loc_system_->Init(config_path_, false, false)) {
                LOG(ERROR) << "LocSystem 初始化失败";
                result->message = "LocSystem initialization failed";
                goal_handle->abort(result);
                stopLocalization();  // 清理未成功启动的定位系统
                return;
            }

            // 锁定初始位姿状态，防止等待 INS 期间 INS 回调抢占初始化
            // SwitchMapAndReinit 内部会根据 INS 是否有效选择位姿：
            //   INS 有效 -> 用 INS 位姿；INS 无效 -> 用 goal 传入位姿
            loc_system_->LockInitPose();

            // 等待接收 INS 数据（最多 5 秒），SwitchMapAndReinit 内部会优先使用 INS 位姿
            LOG(INFO) << "等待 INS 数据...";
            for (int i = 0; i < 100 && rclcpp::ok() && running_; ++i) {
                loc_system_->SpinSome();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            // 使用当前已加载地图重新初始化位姿（地图不变，仅初始化位姿）
            bool success = loc_system_->SwitchMapAndReinit(map_path, init_pose);
            if (!success) {
                LOG(ERROR) << "定位重新初始化失败，地图: " << map_path;
                result->message = "Localization reinitialization failed";
                goal_handle->abort(result);
                stopLocalization();
                return;
            }

            // 启动定位后台线程
            {
                std::lock_guard<std::mutex> lock(mode_mutex_);
                localization_running_ = true;
                current_mode_ = Mode::LOCALIZATION;
            }
            if (localization_thread_.joinable()) {
                localization_thread_.join();
            }
            localization_thread_ = std::thread(&ModeSwitchNode::locLoop, this, loc_system_);
            LOG(INFO) << "定位模块已重新启动（地图未变更: " << loaded_map_path
                      << "），等待定位收敛...";

            // 3. 监控定位状态，定位正常且 NDT 匹配置信度大于阈值时反馈重定位成功
            //    重定位与定位模块分离：重定位仅负责监控，定位模块在后台持续运行
            // 读取 NDT 匹配置信度阈值
            double min_confidence = 1.8;  // 默认阈值
            try {
                YAML_IO yaml(config_path_);
                min_confidence = yaml.GetValue<double>("lidar_loc", "min_init_confidence");
                LOG(INFO) << "重定位 NDT 匹配置信度阈值: " << min_confidence;
            } catch (const std::exception& e) {
                LOG(WARNING) << "读取 min_init_confidence 失败，使用默认值 " << min_confidence << ": " << e.what();
            }

            rclcpp::Rate loop_rate(100);  // 100 Hz
            auto last_feedback_time = this->now();
            auto reloc_start_time = this->now();
            constexpr double kRelocTimeoutSec = 60.0;  // 重定位超时时间

            while (rclcpp::ok() && running_) {
                if (goal_handle->is_canceling()) {
                    LOG(INFO) << "Relocation goal 被取消";
                    result->message = "Relocation cancelled";
                    goal_handle->canceled(result);
                    return;  // 定位模块继续运行
                }

                auto now = this->now();
                if ((now - last_feedback_time).seconds() >= 1.0) {
                    feedback->state = nav2_interface::msg::NavState::NAV_STA_TASK_EXEC_DOING;
                    goal_handle->publish_feedback(feedback);
                    last_feedback_time = now;
                }

                // 检查 NDT 初始化是否成功 且 置信度大于阈值
                // 在锁内操作 loc_system_ shared_ptr，避免与 /map/switch 的 stopLocalization 竞争
                bool loc_ok = false;
                double ndt_confidence = 0.0;
                {
                    std::lock_guard<std::mutex> lock(mode_mutex_);
                    if (loc_system_ && loc_system_->IsNDTInited()) {
                        ndt_confidence = loc_system_->GetConfidence();
                        loc_ok = true;
                    }
                }
                if (loc_ok) {
                    LOG(INFO) << "NDT初始化成功，当前 NDT 匹配置信度: " << ndt_confidence
                              << "，阈值: " << min_confidence;
                    if (ndt_confidence >= min_confidence) {
                        LOG(INFO) << "重定位成功，NDT 匹配置信度 " << ndt_confidence << " >= " << min_confidence;
                        result->message = "Relocation success";
                        goal_handle->succeed(result);
                        return;  // 定位模块继续运行，重定位流程结束
                    }
                }

                // 超时检查
                if ((now - reloc_start_time).seconds() >= kRelocTimeoutSec) {
                    double confidence = 0.0;
                    bool ndt_inited = false;
                    {
                        std::lock_guard<std::mutex> lock(mode_mutex_);
                        if (loc_system_) {
                            confidence = loc_system_->GetConfidence();
                            ndt_inited = loc_system_->IsNDTInited();
                        }
                    }
                    LOG(ERROR) << "重定位超时（" << kRelocTimeoutSec << "秒），NDT初始化: "
                               << (ndt_inited ? "成功" : "未完成")
                               << "，NDT 匹配置信度: " << confidence << "，阈值: " << min_confidence;
                    result->message = "Relocation timeout, NDT confidence not reached threshold";
                    goal_handle->abort(result);
                    return;
                }

                loop_rate.sleep();
            }

            // 节点退出或 ROS 关闭
            LOG(WARNING) << "重定位未完成，节点退出";
            result->message = "Relocation interrupted by node shutdown";
            goal_handle->abort(result);
        } catch (const std::exception& e) {
            LOG(ERROR) << "Relocation 执行异常: " << e.what();
            result->message = std::string("Exception: ") + e.what();
            goal_handle->abort(result);
            stopLocalization();
        }
    }
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = google::INFO;

    rclcpp::init(argc, argv);
    auto node = std::make_shared<ModeSwitchNode>();

    // 设置实时调度策略 SCHED_FIFO，优先级 99
    /*
    struct sched_param param;
    param.sched_priority = 99;
    if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
        LOG(ERROR) << "sched_setscheduler(SCHED_FIFO, 99) failed: " << strerror(errno);
    } else {
        LOG(INFO) << "已设置 SCHED_FIFO 优先级 99";
    }

    // 绑定 CPU 4,5,6
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(4, &cpuset);
    CPU_SET(5, &cpuset);
    CPU_SET(6, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        LOG(ERROR) << "sched_setaffinity(CPU 4,5,6) failed: " << strerror(errno);
    } else {
        LOG(INFO) << "已绑定 CPU 4,5,6";
    } */

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}