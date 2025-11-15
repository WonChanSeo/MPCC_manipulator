cmake_minimum_required(VERSION 3.10)
project(MPCC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA support option
option(USE_CUDA "Enable CUDA GPU acceleration for NN inference" OFF)

if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA GPU acceleration: ON")
  message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
  add_definitions(-DUSE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # Set CUDA architecture
  set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3090 compute capability (sm_86)

  # Set CUDA compiler flags
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

# FlexFloat option for NN inference precision testing
option(NN_USE_FLEXFLOAT "Enable FlexFloat precision control for NN inference" OFF)
set(NN_FF_EXPONENT_BITS "8" CACHE STRING "FlexFloat exponent bits for NN")
set(NN_FF_MANTISSA_BITS "23" CACHE STRING "FlexFloat mantissa bits for NN")

if(NN_USE_FLEXFLOAT)
  message(STATUS "NN FlexFloat precision control: ON")
  message(STATUS "  NN_FF_EXPONENT_BITS: ${NN_FF_EXPONENT_BITS}")
  message(STATUS "  NN_FF_MANTISSA_BITS: ${NN_FF_MANTISSA_BITS}")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/External/flexfloat/include)
  add_definitions(-DNN_USE_FLEXFLOAT)
  add_definitions(-DNN_FF_exponent_bits=${NN_FF_EXPONENT_BITS})
  add_definitions(-DNN_FF_mantissa_bits=${NN_FF_MANTISSA_BITS})
endif()

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/External/Eigen/cmake")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/External/osqp_eigen/lib")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/External/osqp/lib")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/External/rbdl-orb/lib")

option(USE_CVPLOT "Use cv-plot library" OFF)
MESSAGE( STATUS "Using CvPlot: " ${USE_CVPLOT} )

add_definitions(-DBUILD_DIRECTORY=\"${CMAKE_SOURCE_DIR}\")

include_directories(.)
include_directories(External/json/include)
include_directories(/home/mms-wonchan/MPCC_manipulator/cpp/External/Eigen)

set(EIGEN3_INCLUDE_DIR "/home/mms-wonchan/MPCC_manipulator/cpp/External/Eigen")
include_directories(${EIGEN3_INCLUDE_DIR})


find_package(Eigen3 REQUIRED)

message(STATUS "Eigen3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")
find_package(RBDL REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling support")
    add_definitions(-DUSE_OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(
        ${EIGEN3_INCLUDE_DIR}
        ${RBDL_INCLUDE_DIR}
        include
)


if(NN_USE_FLEXFLOAT)
  set(FLEXFLOAT_NN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/External/flexfloat/src/flexfloat.c)
endif()

# CUDA kernel sources
if(USE_CUDA)
  set(CUDA_KERNEL_SRC src/Constraints/EnvCollision/cuda_nn_kernels.cu)
endif()

add_library(MPCC_LIB SHARED
        src/types.cpp
        src/Model/robot_model.cpp
        src/Constraints/SelfCollision/SelfCollisionModel.cpp
        src/Constraints/EnvCollision/EnvCollisionModel.cpp
        src/Interfaces/osqp_interface.cpp
        src/Interfaces/solver_interface.cpp
        src/Model/model.cpp
        src/Model/integrator.cpp
        src/Spline/cubic_spline.cpp
        src/Spline/cubic_spline_rot.cpp
        src/Spline/arc_length_spline.cpp
        src/Params/params.cpp
        src/Constraints/constraints.cpp
        src/Cost/cost.cpp
        src/Constraints/bounds.cpp
        src/MPC/mpc.cpp
        src/Params/track.cpp
        include/config.h
        include/Model/robot_model.h
        ${FLEXFLOAT_NN_SRC}
        ${CUDA_KERNEL_SRC}
        )

# Set CUDA source file properties
if(USE_CUDA)
  set_source_files_properties(src/Constraints/EnvCollision/cuda_nn_kernels.cu
    PROPERTIES LANGUAGE CUDA)
endif()

target_link_libraries(MPCC_LIB PUBLIC
        ${RBDL_LIBRARY}
        OsqpEigen::OsqpEigen
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(MPCC_LIB PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_CUDA)
    target_link_libraries(MPCC_LIB PUBLIC CUDA::cublas CUDA::cudart)
endif()

add_executable(MPCC_EXE src/main.cpp)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
target_link_libraries(MPCC_EXE MPCC_LIB)

if(USE_CVPLOT)
        find_package (OpenCV REQUIRED)
        include_directories(${OpenCV_INCLUDE_DIRS})
        include_directories(External/cv-plot/lib/include)
        target_link_libraries(MPCC_EXE ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR}/External/cv-plot/lib/lib/libCvPlot.so)
        target_compile_definitions(MPCC_EXE PRIVATE USE_CVPLOT)
endif()

add_executable(MPCC_TEST
        src/Tests/main_test.cpp
        include/Tests/model_integrator_test.h
        include/Tests/robot_model_test.h
        include/Tests/self_collision_test.h
        include/Tests/spline_test.h
        include/Tests/constraints_test.h
        include/Tests/cost_test.h
)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
find_package(Python COMPONENTS Development)
target_include_directories(MPCC_TEST PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(MPCC_TEST PRIVATE MPCC_LIB gtest gtest_main pthread ${Python_LIBRARIES})

find_package(Boost REQUIRED python)
find_package(PythonLibs REQUIRED)
find_package(eigenpy REQUIRED)

include_directories(
        ${Boost_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${eigenpy_INCLUDE_DIRS}
)

add_library(MPCC_WRAPPER SHARED src/MPCC_wrapper.cpp)
target_link_libraries(MPCC_WRAPPER 
        ${Boost_LIBRARIES} 
        ${PYTHON_LIBRARIES} 
        eigenpy::eigenpy
        Eigen3::Eigen 
        MPCC_LIB
)
set_target_properties(MPCC_WRAPPER PROPERTIES PREFIX "")