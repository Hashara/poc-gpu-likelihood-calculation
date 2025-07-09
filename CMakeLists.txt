cmake_minimum_required(VERSION 3.26)
project(poc-gpu-likelihood-calculations)

set(CMAKE_CXX_STANDARD 17)

set(SRC
        main.cpp
        helper/Matrix.cpp
        helper/Matrix.h
        helper/MatrixOp.h
        helper/MatrixOpCPU.cpp
        helper/MatrixOpCPU.h
        helper/MatrixOpFactory.cpp
        helper/MatrixOpFactory.h
)
######################################################
# setup compilers
######################################################

set(NVHPC "FALSE") # NVIDIA HPC SDK compiler
set (GCC "FALSE") # GNU Compiler Collection
set(CLANG "FALSE") # Clang compiler

if (CMAKE_CXX_COMPILER_ID MATCHES "NVHPC") # NVIDIA HPC SDK compiler
    message("Compiler      : NVIDIA HPC SDK Compiler (nvc++)")
    set(NVHPC "TRUE")
    add_definitions(-DUSE_NVHPC)
endif ()

##################################################################
# configure OpenACC compilation
##################################################################
if (USE_OPENACC STREQUAL "ON")
    message("OpenACC       : Yes")
    if (GCC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenacc")
        list(APPEND SRC
                helper/MatrixOpOpenACC.cpp
                helper/MatrixOpOpenACC.h
        )
    elseif (NVHPC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc")
        list(APPEND SRC
                helper/MatrixOpOpenACC.cpp
                helper/MatrixOpOpenACC.h
        )
    endif()
else()
    message("OpenACC       : NONE")
endif()

add_executable(gpulcal ${SRC})
