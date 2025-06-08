cmake_minimum_required(VERSION 3.5.0)
project(evolutionSimulator)

FILE(GLOB MyCppSources ./src/*.cpp ./src/*.h)

find_package(OpenMP REQUIRED)

find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(evolutionSimulator ${MyCppSources})
target_link_libraries(evolutionSimulator 
    PRIVATE
      OpenMP::OpenMP_CXX
      ${OpenCV_LIBS}
)