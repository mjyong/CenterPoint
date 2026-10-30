cmake_minimum_required(VERSION 3.8)
project(common_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/EulerAngles.msg"
  "msg/Header.msg"
  "msg/MapType.msg"
  "msg/Point.msg"
  "msg/Pose.msg"
  "msg/PoseWithCovariance.msg"
  "msg/Quaternion.msg"
  "msg/Size3.msg"
  "msg/Time.msg"
  "msg/Vector3.msg"
  "msg/TwistWithCovariance.msg"
  "msg/PointField.msg"
  "msg/GeograpicPose.msg"
  "msg/GeometryPose.msg"
  "msg/BoundingBox2D.msg"

  "srv/Example.srv"
  DEPENDENCIES std_msgs geometry_msgs# Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  # set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  # set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
