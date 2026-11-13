cmake_minimum_required(VERSION 3.5)
project(seres_proto)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# generate protobuf
if(DEFINED ENV{SOC_PLATFORM} AND "$ENV{SOC_PLATFORM}" STREQUAL "QCOM_6490")
  set(PROTOBUF_PROTOC_EXECUTABLE /usr/local/bin/protoc)
  set(PROTOBUF_ANDROID_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../third/protobuf/android)
else()
  find_package(Protobuf REQUIRED)
endif()
set(PROTODIR_HDRS ${PROJECT_SOURCE_DIR}/.temp/include/proto/srcp)
set(PROTODIR_TEMP ${PROJECT_SOURCE_DIR}/.temp/proto)

file(REMOVE_RECURSE  ${PROTODIR_HDRS} ${PROTODIR_TEMP})
file(MAKE_DIRECTORY  ${PROTODIR_HDRS} ${PROTODIR_TEMP})

set(PROTODIR "srcp")
file(GLOB PROTO_FILES "${PROTODIR}/*.proto")

list(APPEND PROTO_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/${PROTODIR})
foreach(FILE ${PROTO_FILES})
  get_filename_component(FILE_NAME ${FILE} NAME_WE)
  set(PROTO_HDR "${FILE_NAME}.pb.h")

  execute_process(
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTO_FLAGS} --cpp_out=${PROTODIR_TEMP} ${FILE}
    RESULT_VARIABLE PROTOC_RESULT
  )
  if(NOT PROTOC_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to generate files from ${FILE}")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} -E copy 
    "${PROTODIR_TEMP}/${PROTO_HDR}"
    "${PROTODIR_HDRS}/${PROTO_HDR}")
endforeach()

file(GLOB SRC_FILES
  "${PROTODIR_TEMP}/*.pb.cc"
)
# set(LIB_PB_SRCP "seres_srcp")
set(LIB_PB_SRCP "seres_proto")
add_library(
  ${LIB_PB_SRCP}
  SHARED
  ${SRC_FILES}
)

if(DEFINED ENV{SOC_PLATFORM} AND "$ENV{SOC_PLATFORM}" STREQUAL "QCOM_6490")
  target_include_directories(
    seres_proto
    PUBLIC
    $<BUILD_INTERFACE:${PROTODIR_HDRS}>
    $<BUILD_INTERFACE:${PROTOBUF_ANDROID_ROOT}/include>
  )
  target_link_libraries(
    seres_proto
    ${PROTOBUF_ANDROID_ROOT}/lib/libprotobuf.a
    log
  )
endif()

install(
  DIRECTORY .temp/include/
  DESTINATION include/seres_interface
)

# 新增：安装 proto 源文件到同一个 include 目录
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/srcp/
  DESTINATION include/seres_interface/proto/srcp
  FILES_MATCHING PATTERN "*.proto"
)

install(TARGETS ${LIB_PB_SRCP}
  EXPORT ${LIB_PB_SRCP}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
ament_export_targets(${LIB_PB_SRCP} HAS_LIBRARY_TARGET)
ament_export_include_directories(include/seres_interface)
ament_export_libraries(${LIB_PB_SRCP})

ament_package()
