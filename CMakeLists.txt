cmake_minimum_required(VERSION 3.5.0)
project(evolutionSimulator)

FILE(GLOB MyCppSources ./src/*.cpp)
FILE(GLOB MyCppHeaders ./src/*.h)

find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(evolutionSimulator ${MyCppSources} ${MyCppHeaders})
target_link_libraries(evolutionSimulator ${OpenCV_LIBS} )