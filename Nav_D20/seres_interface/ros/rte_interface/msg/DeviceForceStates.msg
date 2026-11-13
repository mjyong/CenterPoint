 #   rclcpp::QoS qos_profile(50); // 深度为50
 #   qos_profile.keep_last(30);   // 保持最近的30条消息
 #   qos_profile.reliable();      // 使用可靠的传输方式
std_msgs/Header header
DeviceForceState[] states  # 0-左臂六维力状态，1-右臂六维力状态