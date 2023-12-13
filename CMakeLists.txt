cmake_minimum_required(VERSION 3.20)
project(patato_cplusplus_onnx)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(patato_cplusplus_onnx main.cpp)

target_link_libraries(patato_cplusplus_onnx PRIVATE ${OpenCV_LIBS})

set_target_properties(patato_cplusplus_onnx PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)