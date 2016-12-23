cmake_minimum_required(VERSION 2.8)
project (XOR_ANN)

set(CMAKE_CXX_STANDARD 11)

include_directories("../include/")
file(GLOB SOURCES "../src/*.cpp")
add_executable(XOR_ANN ${SOURCES})
