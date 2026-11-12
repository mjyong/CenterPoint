add_library(${PROJECT_NAME}.libs SHARED
        common/nav_state.cc
        common/constant.h
        common/options.cc
        common/params.cc

        io/yaml_io.cc
        io/file_io.cc

        core/lio/eskf.cc
        core/lio/laser_mapping.cc
        core/lio/pointcloud_preprocess.cc
        core/super_lio_frontend/laser_mapping_superlio.cc
        core/loop_closing/loop_closing.cc

        core/g2p5/g2p5_map.cc
        core/g2p5/g2p5.cc
        core/g2p5/g2p5_subgrid.cc

        core/maps/tiled_map.cc
        core/maps/tiled_map_chunk.cc

        core/localization/localization.cpp

        utils/timer.cc
        utils/pointcloud_utils.cc

        # ui/pangolin_window.cc
        # ui/pangolin_window_impl.cc
        ui/rviz_visualizer.cc
        # ui/ui_car.cc
        # ui/ui_cloud.cc
        # ui/ui_trajectory.cc

        core/system/slam.cc
        core/system/loc_system.cc

        wrapper/bag_io.cc
)

ament_target_dependencies(
        ${PROJECT_NAME}.libs
        "rclcpp"
        "std_msgs"
        "geometry_msgs"
        "sensor_msgs"
        "nav_msgs"
        "std_srvs"
        "message_filters"
        "pcl_conversions"
        "tf2_ros"
        "tf2"
        "novatel_oem7_msgs"
        "kygi665_msgs"
        livox_ros_driver2
        basic
        nav2_interface
        rhr_interface
)

add_dependencies(${PROJECT_NAME}.libs

        ${PROJECT_NAME}__rosidl_typesupport_cpp
)

include_directories(${PROJECT_SOURCE_DIR}/src/core/miao)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core/super_lio_frontend/include)
add_subdirectory(core/miao)
add_subdirectory(core/super_lio_frontend)

target_link_libraries(${PROJECT_NAME}.libs
        ${third_party_libs}
        ${PCL_LIBRARIES}
        miao.core
        miao.utils
        super_lio_frontend_lib
        basic::basic
        ${PROJECT_NAME}__rosidl_typesupport_cpp
)

install(TARGETS ${PROJECT_NAME}.libs miao.core miao.utils
        DESTINATION lib)

add_subdirectory(app)
