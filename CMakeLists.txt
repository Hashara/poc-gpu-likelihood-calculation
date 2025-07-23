cmake_minimum_required(VERSION 3.26)
project(poc-gpu-likelihood-calculations)

set(CMAKE_CXX_STANDARD 17)

option(USE_OPENACC "Enable OpenACC support" OFF)
option(USE_CUDA     "Enable CUDA support" OFF)
option(USE_CUBLAS   "Enable CUBLAS support" OFF)
option(USE_OPENMP_GPU "Enable OpenMP GPU support" OFF)
option(USE_EIGEN "Use Eigen3 library" OFF)

set(SRC
        main.cpp
        helper/Matrix.cpp
        helper/Matrix.h
        helper/MatrixOp.h
        helper/MatrixOpType.h
        helper/MatrixUtils.h

        alignment/Alignment.cpp
        alignment/Alignment.h
        alignment/Pattern.cpp
        alignment/Pattern.h
        alignment/alignmentIO.cpp
        alignment/alignmentIO.h

        tree/Node.cpp
        tree/Node.h
        tree/Tree.cpp
        tree/Tree.h
        tree/TreeReader.cpp
        tree/TreeReader.h
        tree/TreeUtils.h

        model/Model.cpp
        model/Model.h
        model/ModelJC.cpp
        model/ModelJC.h
        tree/LikelihoodCalculator.cpp
        tree/LikelihoodCalculator.h
        helper/MatrixOpDispatcher.h
        helper/logger/Logger.cpp
        helper/logger/Logger.h
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

##################################################################
# configure CUBLAS
##################################################################

if (USE_CUBLAS)
    message("CUBLAS        : Yes")
    if (NVHPC)
        find_package(CUDA REQUIRED)
        if (CUDA_FOUND)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUBLAS")
            include_directories(${CUDA_INCLUDE_DIRS})
            link_directories(${CUDA_LIBRARY_DIRS})
            list(APPEND SRC
                    helper/MatrixOpCUBLAS.cpp
                    helper/MatrixOpCUBLAS.h
            )
        else()
            message(FATAL_ERROR "CUDA not found. Either install it or rerun cmake with -DCUDA_ROOT=<installed_cuda_dir>")
        endif()
    else()
        message(FATAL_ERROR "CUBLAS requires CUDA support. Enable CUDA with -DUSE_CUDA=ON")
    endif()
else()
    message("CUBLAS        : NONE")
endif()

##################################################################
# configure Eigen
##################################################################
# Find Eigen3 library
if (USE_EIGEN)
    message("Eigen3        : Yes")
    if (NOT EIGEN3_INCLUDE_DIR)
        find_package(Eigen3)
        if(NOT EIGEN3_FOUND)
            message(FATAL_ERROR "Eigen3 library not found. Either install it or rerun cmake with -DEIGEN3_INCLUDE_DIR=<installed_eigen3_dir>")
        else()
            add_definitions(-DUSE_EIGEN)
        endif()
    endif()
    add_definitions(-I${EIGEN3_INCLUDE_DIR})
else()
    message("Eigen3        : NONE")
    list(APPEND SRC
            helper/MatrixOpCPU.cpp
            helper/MatrixOpCPU.h)
endif ()

##################################################################

add_executable(gpulcal ${SRC})


##################################################################
# Include the test library
##################################################################
option(USE_GTEST "Use Google Test library (gtest)" OFF)
if (USE_GTEST)
    message("Use Google Test library: ON")
    add_definitions(-DUSE_GTEST)
    if (NOT GTEST_ROOT)
        find_package(GTest REQUIRED)
        if (GTEST_FOUND)
            include_directories(${GTEST_INCLUDE_DIRS})
            link_directories(${GTEST_LIBRARY_DIRS})
        else()
            message(FATAL_ERROR "Google Test library not found. Either install it or rerun cmake with -DGTEST_ROOT=<installed_gtest_dir>")
        endif()
    else()
        include_directories(${GTEST_ROOT}/include)
        link_directories(${GTEST_ROOT}/lib)
    endif()
    if (GTEST_FOUND)
        message("Google Test library found: ${GTEST_LIBRARIES}")
        add_definitions(-DGTEST_FOUND)
        # --- GoogleTest ----------------------------------------------------------------
        include(FetchContent)

        # Always good to pin an exact tag / commit for reproducibility
        FetchContent_Declare(
                googletest
                URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

        FetchContent_MakeAvailable(googletest)   # defines gtest & gtest_main targets


        # --- Unit-test target ----------------------------------------------------------
        enable_testing()

        set(TEST_SRC
                helper/Matrix.cpp
                helper/MatrixOpCPU.cpp
        )

        add_library(matrixlib STATIC ${TEST_SRC})
        target_include_directories(matrixlib PUBLIC ${PROJECT_SOURCE_DIR}/helper)

        # Link it with test executable
        add_executable(test_matrix test/helper/testMatrix.cpp)

        target_link_libraries(test_matrix
                PRIVATE
                matrixlib
                gtest
                gtest_main
        )

        gtest_discover_tests(test_matrix)

        add_test(NAME RunAllTests COMMAND test_matrix)

    else()
        message(FATAL_ERROR "Google Test library not found. Either install it or rerun cmake with -DGTEST_ROOT=<installed_gtest_dir>")
    endif()


endif ()