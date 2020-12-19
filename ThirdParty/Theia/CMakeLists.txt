cmake_minimum_required(VERSION 2.8)
project( Test )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable( Test main.cpp )
target_link_libraries( Test ${OpenCV_LIBS} ${CERES_LIBRARIES})