cmake_minimum_required(VERSION 3.1)
project(find_points)
set (CMAKE_CXX_STANDARD 17)
set("OpenCV_DIR" "/usr/local/Cellar/opencv/4.3.0_5/include/opencv4")
find_package(OpenCV REQUIRED)

add_executable(main
    "find_points.cpp"
)

# link the new hello_library target with the hello_binary target
target_link_libraries( main
     ${OpenCV_LIBS}
)
