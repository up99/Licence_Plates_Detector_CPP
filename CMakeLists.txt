cmake_minimum_required(VERSION 3.10)
project(TestTask)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
# Find OpenCV
find_package(OpenCV REQUIRED PATHS "C:/opencv/build" NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(modules/inference)
include_directories(modules/config)
include_directories(modules/genpriors)
include_directories(modules/postprocessing)
include_directories(modules/visualization)


add_executable(main main.cpp 
                    modules/inference/inference.cpp
                    modules/genpriors/genpriors.cpp
                    modules/postprocessing/postprocessing.cpp
                    modules/visualization/visualization.cpp)


target_link_libraries(main ${OpenCV_LIBS})