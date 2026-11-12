cmake_minimum_required(VERSION 3.8)
project(nav_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(common_interface REQUIRED)
find_package(rosidl_default_generators REQUIRED)

#自定义消息接口添加部分
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TakePhoto.msg"
  "msg/LinearScan.msg"
  "msg/AlgoDiagInfo.msg"
  "msg/AlgorithmDiagnosticStatus.msg"
  "msg/AudioCtrl.msg"
  "msg/AudioVideoStreamCtrl.msg"
  "msg/ActionSchedule.msg"
  "msg/CostMap.msg"  
  "msg/CurrentPTZ.msg"  
  "msg/DetectionObject.msg"  
  "msg/DetectionSegArray.msg"      
  "msg/FineTuningPTZ.msg"      
  "msg/LidarPreprocessState.msg"      
  "msg/MapData2D.msg"      
  "msg/MapMetaData.msg"      
  "msg/MapState.msg"      
  "msg/Mode.msg"      
  "msg/NavAlgorithmDiagnosticStatus.msg"      
  "msg/NavMode.msg"      
  "msg/NavPatrolTaskCtrl.msg"      
  "msg/NavPedestrianStandard.msg"      
  "msg/NavPlannerMotion.msg"      
  "msg/NavPtzAngleCtrl.msg"
  "msg/NavPtzCamCtrl.msg"
  "msg/NavPtzTakePhotoCtrl.msg"      
  "msg/NavRemoteControllerRequest.msg"      
  "msg/NavSensorCamera.msg"      
  "msg/NavSkillStatus.msg"      
  "msg/Odometry.msg"      
  "msg/OdometryState.msg"      
  "msg/PatrolTaskCtrl.msg"      
  "msg/PedestrianStandard.msg"
  "msg/PlanningSchedule.msg"   
  "msg/PlaningProcTask.msg"      
  "msg/PlaningState.msg"
  "msg/PlannerMotion.msg"  
  "msg/Point3D.msg"    
  "msg/PointCloud2.msg"
  "msg/PtzAngleCtrl.msg"
  "msg/PtzCamCtrl.msg"
  "msg/PtzTakePhotoCtrl.msg"       
  "msg/RealtimeVoice.msg"      
  "msg/RemoteControllerRequest.msg"      
  "msg/RemoteTaskResponse.msg"      
  "msg/RemoteTasksCtrl.msg"      
  "msg/SegmentArray.msg"      
  "msg/SkillStatus.msg"      
  "msg/TargetPTZ.msg"      
  "msg/Task.msg"     
  "msg/TaskExecuteSwitch.msg"  
  "msg/TraversabilityGrid.msg"  
  "msg/UploadTask.msg"  
  "msg/UploadTaskList.msg"  
  "msg/VisualPerceptionState.msg"  
  "msg/WholePointCloud2.msg"  
  "msg/CamTemMatrix.msg"
  "msg/DetSegResults.msg"
  "msg/Visual2D.msg"
  "msg/Header.msg"
  "msg/PTZConfirm.msg"
  
  "msg/Point2D.msg"
  "msg/KeyPoint.msg"
  "msg/Area.msg"
  "msg/SemanticMap.msg"
  "msg/Pose6D.msg"
  "srv/SemanticMapSrv.srv"

  "msg/PtzRGBImgData.msg"
  "msg/PtzInfrardImgData.msg"
  "msg/NavInfraredResult.msg"
  "msg/AbnormalCondition.msg"
  "msg/DiagInfo.msg"

  DEPENDENCIES
    "common_interface"
    "std_msgs"
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
