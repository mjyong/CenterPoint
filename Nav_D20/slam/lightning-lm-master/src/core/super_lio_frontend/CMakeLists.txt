add_definitions(-DROOT="${CMAKE_CURRENT_SOURCE_DIR}/")

add_library(super_lio_frontend_lib STATIC
  lio/ESKF.cpp
  lio/params.cpp
)

target_include_directories(super_lio_frontend_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(super_lio_frontend_lib PUBLIC
  basic::basic
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  glog
  yaml-cpp
)
