cmake_minimum_required(VERSION 2.8)

project(g2oCurveFitting)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")



# list(APPEND CMAKE_MODULE_PATH ~/vslam_learn/slambook2/3rdparty/g2o/cmake_modules) 
# list(APPEND CMAKE_MODULE_PATH ~/vslam_learn/slambook2/3rdparty/g2o/cmake_modules)
# set(g2o_ROOT /usr/local/include/g2o)
# list( APPEND CMAKE_MODULE_PATH ~/vslam_learn/slambook2/ch6/cmake )
# LIST(g2o_ROOT /usr/local/include/g2o/cmake_modules)
# SET(g2o_ROOT /usr/local/include/g2o)
# FIND_PACKAGE(g2o REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# g2o
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# eigen
include_directories("/usr/include/eigen3")


add_executable(g2oCurveFitting g2oCurveFitting.cpp)
# MESSAGE(STATUS ${g2o_CORE_LIBRARY})
# MESSAGE(STATUS ${g2o_STUFF_LIBRARY})
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} ${G2G_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})