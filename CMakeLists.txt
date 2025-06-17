cmake_minimum_required(VERSION 3.18)

project(evolutionSimulator)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(evolutionSimulator
  src/simulator.cpp
  src/simulator.hpp
  src/brain.cpp
  src/brain.hpp
  src/video_writer.cpp
  src/video_writer.hpp
)
target_link_libraries(evolutionSimulator 
    PRIVATE
      OpenMP::OpenMP_CXX
      ${OpenCV_LIBS}
)