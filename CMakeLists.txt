cmake_minimum_required(VERSION 3.5.0)
project(evolution_simulator)

FILE(GLOB MyCppSources ./src/*.cpp)
FILE(GLOB MyCppHeaders ./src/*.h)

set(OpenCV_DIR C:/opencv/opencv/build)

find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(evolution_simulator ${MyCppSources} ${MyCppHeaders})
target_link_libraries(evolution_simulator ${OpenCV_LIBS} )