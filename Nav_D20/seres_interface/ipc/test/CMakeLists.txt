cmake_minimum_required(VERSION 3.18)

project(UtilsTest)
set(CMAKE_CXX_STANDARD 17)
message("GOOGLE_TEST_SOURCE:" ${GOOGLE_TEST_SOURCE})
add_compile_options(-g)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Install path" FORCE)
endif()

message("CMAKE_INSTALL_PREFIX:" ${CMAKE_INSTALL_PREFIX})

add_definitions(-DUNIT_TEST)

include_directories(${GOOGLE_TEST_SOURCE}/googletest/include
                    ${GOOGLE_TEST_SOURCE}/googlemock/include
                    ${CMAKE_SOURCE_DIR}/../utils/)

link_directories(build_googletest/lib)

#test timer_manager.cpp
add_executable(utils_test
            main.cpp
            seres_string_test.cpp
            seres_vector_test.cpp)

target_link_libraries(utils_test gtest)

install(TARGETS utils_test
        DESTINATION ${CMAKE_INSTALL_PREFIX}/package/nav/module_utils/bin
    )