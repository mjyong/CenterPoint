/**
 * 揽沃雷达/超声波传感器健康监测节点（独立节点，不依赖定位库）
 *
 * 监测话题:
 *   /robot/topic/rdap/front_point_cloud2  sensor_msgs/PointCloud2  头部激光雷达(NN=33)
 *   /robot/topic/rdap/rear_point_cloud2   sensor_msgs/PointCloud2  尾部激光雷达(NN=34)
 *   /robot/topic/rdap/radar_front_left    sensor_msgs/Range        头部超声波雷达(NN=31)
 *   /robot/topic/rdap/radar_front_right   sensor_msgs/Range        尾部超声波雷达(NN=32)
 *
 * 上报告警 topic /rhr/topic/alarm_info (QoS 50 reliable, 状态去重):
 *   0x0103NN04 点云数据骤降: 连续3帧点云计数少于正常值(EMA基线)的50%, 连续10帧正常恢复 (仅激光雷达)
 *   0x0103NN08 帧间隔抖动:   帧间隔大于理论值(自适应学习)的150%, 连续5帧触发, 连续10帧正常恢复;
 *                            数据中断超过5倍理论间隔(最少3秒)立即触发;
 *                            启动宽限期(10秒)后从未收到数据也立即触发(由看门狗定时器判定, 不依赖回调)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <unordered_map>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/range.hpp>

#include "rhr_interface/msg/alarm_info.hpp"

namespace {

using AlarmInfo = rhr_interface::msg::AlarmInfo;

uint64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

/// 单个传感器的监测状态
struct SensorState {
    std::string name;            // 告警名前缀, 如 "头部激光雷达"
    uint32_t cloud_drop_id = 0;  // 点云骤降告警ID, 0=不适用(超声波)
    uint32_t jitter_id = 0;      // 帧间隔抖动告警ID

    // 帧间隔
    double last_recv_sec = 0.0;      // 上一帧到达时刻(系统秒)
    double expected_interval = 0.0;  // 理论帧间隔(自适应学习)
    int learn_cnt = 0;               // 学习期已收帧数
    double learn_sum = 0.0;          // 学习期间隔累计
    int jitter_cnt = 0;              // 连续抖动帧计数(去抖)
    bool jitter_active = false;      // 当前是否处于抖动告警中

    // 点云数量(仅激光雷达)
    double cloud_baseline = 0.0;  // 点云数目基线(EMA)
    int drop_cnt = 0;             // 连续骤降帧计数
    bool drop_active = false;     // 当前是否处于骤降告警中
};

}  // namespace

class RdapSensorMonitor : public rclcpp::Node {
   public:
    explicit RdapSensorMonitor(const rclcpp::NodeOptions& options) : Node("rdap_sensor_monitor", options) {
        rclcpp::QoS qos(50);
        qos.reliable();
        alarm_pub_ = this->create_publisher<AlarmInfo>("/rhr/topic/alarm_info", qos);

        // (名称, 点云骤降ID, 帧间隔抖动ID) — NN 编号见《事件和警告.xlsx》
        // 33=头部激光雷达 34=尾部激光雷达 31=头部超声波 32=尾部超声波
        sensors_["front_lidar"] = {"头部激光雷达", 0x01033304u, 0x01033308u};
        sensors_["rear_lidar"] = {"尾部激光雷达", 0x01033404u, 0x01033408u};
        sensors_["front_radar"] = {"头部超声波雷达", 0u, 0x01033108u};
        sensors_["rear_radar"] = {"尾部超声波雷达", 0u, 0x01033208u};

        cloud_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions sub_opts;
        sub_opts.callback_group = cloud_cb_group_;

        auto& nh = *this;
        pc_front_sub_ = nh.create_subscription<sensor_msgs::msg::PointCloud2>(
            "/robot/topic/rdap/front_point_cloud2", qos,
            [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                OnCloud(sensors_["front_lidar"], msg->width * msg->height);
            },
            sub_opts);
        pc_rear_sub_ = nh.create_subscription<sensor_msgs::msg::PointCloud2>(
            "/robot/topic/rdap/rear_point_cloud2", qos,
            [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                OnCloud(sensors_["rear_lidar"], msg->width * msg->height);
            },
            sub_opts);
        range_front_sub_ = nh.create_subscription<sensor_msgs::msg::Range>(
            "/robot/topic/rdap/radar_front_left", qos,
            [this](sensor_msgs::msg::Range::SharedPtr) { OnFrame(sensors_["front_radar"]); }, sub_opts);
        range_rear_sub_ = nh.create_subscription<sensor_msgs::msg::Range>(
            "/robot/topic/rdap/radar_front_right", qos,
            [this](sensor_msgs::msg::Range::SharedPtr) { OnFrame(sensors_["rear_radar"]); }, sub_opts);

        // 1Hz 看门狗: 定时扫描数据中断(含"启动后从未收到数据"), 不依赖回调触发
        watchdog_timer_ = this->create_wall_timer(std::chrono::seconds(1), [this]() {
            double now = NowSec();
            for (auto& [key, st] : sensors_) {
                if (st.last_recv_sec <= 0.0) {
                    bool never_recv = (now - start_sec_) > kStartupGraceSec;
                    if (never_recv) {
                        st.jitter_cnt = 5;
                    }
                    SetJitterAlarm(st, never_recv);
                    continue;
                }
                double timeout = std::max(5.0 * st.expected_interval, 3.0);
                bool lost = (now - st.last_recv_sec) > timeout;
                if (lost) {
                    st.jitter_cnt = 5;  // 立即达到去抖阈值
                }
                SetJitterAlarm(st, lost);
            }
        });

        RCLCPP_INFO(this->get_logger(), "rdap_sensor_monitor 已启动, 监测 4 个雷达话题");
    }

   private:
    static constexpr double kStartupGraceSec = 10.0;  // 启动宽限期: 该时间内未收到数据不算缺失

    static double NowSec() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    /// 帧到达: 更新帧间隔自适应基线并判定帧间隔抖动
    void OnFrame(SensorState& st) {
        double now = NowSec();
        if (st.last_recv_sec <= 0.0) {
            st.jitter_cnt = 0;
            SetJitterAlarm(st, false);
            st.last_recv_sec = now;
            return;
        }
        double interval = now - st.last_recv_sec;
        if (st.learn_cnt < 30) {
            // 学习期: 取平均作为理论帧间隔, 不判定告警
            // 中断后的恢复帧(间隔过大)不计入样本, 防止污染理论间隔基线
            if (interval <= 3.0) {
                st.learn_sum += interval;
                ++st.learn_cnt;
                if (st.learn_cnt == 30) {
                    st.expected_interval = st.learn_sum / st.learn_cnt;
                    RCLCPP_INFO(this->get_logger(), "%s 理论帧间隔学习完成: %.3f 秒", st.name.c_str(),
                                st.expected_interval);
                }
            }
            if (st.jitter_active) {
                st.jitter_cnt = 0;
                SetJitterAlarm(st, false);
            }
        } else {
            bool jitter = interval > 1.5 * st.expected_interval;  // 大于理论值150%
            if (jitter) {
                st.jitter_cnt = std::min(st.jitter_cnt + 1, 5);
            } else {
                st.jitter_cnt = std::max(st.jitter_cnt - 1, 0);
                // 正常帧缓慢跟踪理论间隔(频率渐变时不误报)
                st.expected_interval = 0.95 * st.expected_interval + 0.05 * interval;
            }
            SetJitterAlarm(st, st.jitter_cnt >= 5);
        }
        st.last_recv_sec = now;
    }

    /// 点云帧到达: 先做帧间隔判定, 再做点云数量骤降判定
    void OnCloud(SensorState& st, size_t cloud_size) {
        OnFrame(st);
        if (st.cloud_drop_id == 0) {
            return;
        }
        if (st.cloud_baseline <= 0.0) {
            st.cloud_baseline = static_cast<double>(cloud_size);
            return;
        }
        bool dropped = static_cast<double>(cloud_size) < st.cloud_baseline * 0.5;
        if (dropped) {
            st.drop_cnt = std::min(st.drop_cnt + 1, 3);
            if (st.drop_cnt >= 3) {
                RCLCPP_WARN(this->get_logger(), "%s 点云骤降: %zu (基线 %.0f)", st.name.c_str(), cloud_size,
                            st.cloud_baseline);
            }
        } else {
            st.drop_cnt = std::max(st.drop_cnt - 1, 0);
            st.cloud_baseline = 0.95 * st.cloud_baseline + 0.05 * static_cast<double>(cloud_size);
        }
        SetAlarm(st.cloud_drop_id, st.drop_cnt >= 3, st.drop_active,
                 "点云数据骤降",
                 "连续多帧采集到的点云计数少于正常值的50%");
    }

    void SetJitterAlarm(SensorState& st, bool active) {
        SetAlarm(st.jitter_id, active, st.jitter_active, "帧间隔抖动", "帧间隔大于理论值的150%");
    }

    void SetAlarm(uint32_t id, bool active, bool& state, const std::string& name, const std::string& comment) {
        if (state == active) {
            return;
        }
        state = active;
        AlarmInfo msg;
        msg.id = id;
        msg.time = NowMs();
        msg.state = active ? AlarmInfo::ALARM_ON : AlarmInfo::ALARM_OFF;
        msg.level = AlarmInfo::ALARM_GENERAL;
        msg.para = 0;
        msg.comment = comment;
        msg.name = name;
        alarm_pub_->publish(msg);
        RCLCPP_WARN(this->get_logger(), "[%s] %s (id=0x%08X)", name.c_str(), active ? "告警触发" : "告警恢复", id);
    }

    rclcpp::Publisher<AlarmInfo>::SharedPtr alarm_pub_;
    rclcpp::CallbackGroup::SharedPtr cloud_cb_group_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_front_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_rear_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr range_front_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr range_rear_sub_;
    rclcpp::TimerBase::SharedPtr watchdog_timer_;
    std::unordered_map<std::string, SensorState> sensors_;
    double start_sec_ = NowSec();  // 节点启动时刻(用于启动宽限期判定)
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RdapSensorMonitor>(rclcpp::NodeOptions());
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
