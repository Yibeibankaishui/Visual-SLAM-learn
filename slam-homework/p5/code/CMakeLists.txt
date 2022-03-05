cmake_minimum_required(VERSION 2.8)
project(homework5)

set(CMAKE_BUILD_TYPE "Release")
# add_definitions("-DENABLE_SSE")
# set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# find_package(OpenCV 3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)

find_package(Pangolin REQUIRED)
if(USE_UBUNTU_20)
    message("You are using Ubuntu 20.04, fmt::fmt will be linked")
    find_package(fmt REQUIRED)
    set(FMT_LIBRARIES fmt::fmt)
endif()

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        ${Pangolin_INCLUDE_DIRS}
)

add_executable(computeORB computeORB.cpp)
target_link_libraries(computeORB ${OpenCV_LIBS})

add_executable(E2Rt E2Rt.cpp)
target_link_libraries(E2Rt)

add_executable(GN-BA GN-BA.cpp)
target_link_libraries(GN-BA)

add_executable(TrackAlignment TrackAlignment.cpp)
target_link_libraries(TrackAlignment ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${FMT_LIBRARIES})

