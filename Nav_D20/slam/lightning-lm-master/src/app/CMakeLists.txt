add_executable(run_frontend_offline
        run_frontend_offline.cc
)

target_link_libraries(run_frontend_offline
        ${PROJECT_NAME}.libs
        ${third_party_libs}
)

add_executable(run_loop_offline
        run_loop_offline.cc
)

target_link_libraries(run_loop_offline
        ${PROJECT_NAME}.libs
        ${third_party_libs}
)

add_executable(run_slam_offline
        run_slam_offline.cc
)

target_link_libraries(run_slam_offline
        ${PROJECT_NAME}.libs
        ${third_party_libs}
)

add_executable(run_slam_online
        run_slam_online.cc
)

target_link_libraries(run_slam_online
        ${PROJECT_NAME}.libs
        ${third_party_libs}
)

add_executable(run_loc_offline
        run_loc_offline.cc
)

target_link_libraries(run_loc_offline
        ${PROJECT_NAME}.libs
        ${third_party_libs}
)

add_executable(run_loc_online
        run_loc_online.cc
)

target_link_libraries(run_loc_online
        ${PROJECT_NAME}.libs
        ${third_party_libs}
)

# add_executable(test_ui
#         test_ui.cc
# )

# target_link_libraries(test_ui
#         ${PROJECT_NAME}.libs
#         ${third_party_libs}
# )

add_executable(split_map
        split_map.cc
)

target_link_libraries(split_map
        ${PROJECT_NAME}.libs
        ${third_party_libs}
)

add_executable(mode_switch_node
        mode_switch_node.cc
)

target_link_libraries(mode_switch_node
        ${PROJECT_NAME}.libs
        ${third_party_libs}
)

install(TARGETS run_slam_offline run_slam_online run_loc_offline run_loc_online mode_switch_node split_map
        DESTINATION lib/${PROJECT_NAME})
