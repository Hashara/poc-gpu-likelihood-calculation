cmake_minimum_required(VERSION 3.27)
project(poc-gpu-likelihood-calculations)

set(CMAKE_CXX_STANDARD 17)

add_executable(gpulcal main.cpp
        helper.cpp
        helper.h)
