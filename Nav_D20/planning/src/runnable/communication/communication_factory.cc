/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-22 19:32:55
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2024-12-12 15:19:02
 */
#include "src/runnable/communication/communication_factory.h"
#ifdef ENABLE_ROS
#include "src/runnable/communication/ros_bridge/ros_bridge.h"
#include "src/runnable/communication/ros_debug_bridge/ros_debug_bridge.h"
#include "src/runnable/communication/d20_ptz_bridge/d20_ptz_bridge.h"
#include "src/runnable/communication/r11_head_bridge/r11_head_bridge.h"
#endif
namespace planning {
Factory<CommunicationConf::Type, CommunicationBase,
        CommunicationBase *(*)(const CommunicationConf::Conf &conf),
        std::unordered_map<
            CommunicationConf::Type,
            CommunicationBase *(*)(const CommunicationConf::Conf &conf),
            std::hash<int>>>
    CommunicationFactory::communication_factory_;

CommunicationFactory::CommunicationFactory() {
  communication_factory_.Clear();
  // TODO: regist more class

#ifdef ENABLE_ROS
  rclcpp::init(0, nullptr);

  communication_factory_.Register(
      CommunicationConf::RosBridge,
      [](const CommunicationConf::Conf &conf) -> CommunicationBase * {
        return new RosBridge(conf);
      });

  communication_factory_.Register(
      CommunicationConf::RosDebugBridge,
      [](const CommunicationConf::Conf &conf) -> CommunicationBase * {
        return new RosDebugBridge(conf);
      });

  communication_factory_.Register(
      CommunicationConf::D20PtzBridge,
      [](const CommunicationConf::Conf &conf) -> CommunicationBase * {
        return new D20PtzBridge(conf);
      });

  communication_factory_.Register(
      CommunicationConf::R11HeadBridge,
      [](const CommunicationConf::Conf &conf) -> CommunicationBase * {
        return new R11HeadBridge(conf);
      });
#endif
}

std::unique_ptr<CommunicationBase>
CommunicationFactory::CreateCommunication(const CommunicationConf::Conf &conf) {
  return communication_factory_.CreateObject(conf.type(), conf);
}

} // namespace planning
