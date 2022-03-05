cmake_minimum_required(VERSION 2.8)
project(trajectory)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS '-std=c++11')

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")
add_executable(draw_trajectory draw_trajectory.cpp)
target_link_libraries(draw_trajectory ${Sophus_LIBRARIES} ${Pangolin_LIBRARIES})
add_executable(trajectoryError trajectoryError.cpp)
target_link_libraries(trajectoryError ${Sophus_LIBRARIES} ${Pangolin_LIBRARIES})