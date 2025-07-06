cmake_minimum_required(VERSION 3.10)
project(OpenCVTestProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED PATHS "C:/opencv/build" NO_DEFAULT_PATH)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main Main.cpp)

# Link OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS})