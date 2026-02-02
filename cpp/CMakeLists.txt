cmake_minimum_required(VERSION 3.10)
project(MPCC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2 -ffp-contract=off")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# OSQP build options (for BuildOptions struct in config.h)
option(OSQP_USE_FLOAT "OSQP uses single precision float (mirrors OSQP build)" OFF)
option(OSQP_USE_TRUNCATE "OSQP uses truncation mode (mirrors OSQP build)" OFF)
option(OSQP_USE_ROUND_TO_EVEN "OSQP uses round-to-nearest-even mode (same structure as truncation)" OFF)

if(OSQP_USE_FLOAT)
  message(STATUS "OSQP float mode: ON")
  add_definitions(-DOSQP_USE_FLOAT)
endif()

if(OSQP_USE_TRUNCATE)
  message(STATUS "OSQP truncation mode: ON")
  add_definitions(-DOSQP_USE_TRUNCATE)
endif()

if(OSQP_USE_ROUND_TO_EVEN)
  message(STATUS "OSQP round-to-nearest-even mode: ON")
  add_definitions(-DOSQP_USE_ROUND_TO_EVEN)
endif()

# QDLDL Sample Logging option (inherited from OSQP build)
option(QDLDL_ENABLE_SAMPLE_LOGGING "Enable QDLDL sample data logging" OFF)
if(QDLDL_ENABLE_SAMPLE_LOGGING)
  message(STATUS "QDLDL sample logging: ON")
  add_definitions(-DQDLDL_ENABLE_SAMPLE_LOGGING)
endif()

# Truncate mode options (use truncation instead of rounding for float operations)
option(NN_USE_TRUNCATE "Enable truncation mode for NN inference" OFF)
option(NN_USE_ROUND_TO_EVEN "Enable round-to-nearest-even mode for NN inference" OFF)
option(CONSTRAINTS_USE_TRUNCATE "Enable truncation mode for constraints (getEnvcollConstraint)" OFF)
option(CONSTRAINTS_USE_ROUND_TO_EVEN "Enable round-to-nearest-even mode for constraints" OFF)

if(NN_USE_TRUNCATE)
  message(STATUS "NN truncation mode: ON")
  add_definitions(-DNN_USE_TRUNCATE)
endif()

if(NN_USE_ROUND_TO_EVEN)
  message(STATUS "NN round-to-nearest-even mode: ON")
  add_definitions(-DNN_USE_ROUND_TO_EVEN)
endif()

if(CONSTRAINTS_USE_TRUNCATE)
  message(STATUS "Constraints truncation mode: ON")
  add_definitions(-DCONSTRAINTS_USE_TRUNCATE)
endif()

if(CONSTRAINTS_USE_ROUND_TO_EVEN)
  message(STATUS "Constraints round-to-nearest-even mode: ON")
  add_definitions(-DCONSTRAINTS_USE_ROUND_TO_EVEN)
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
        )
target_link_libraries(MPCC_LIB PUBLIC
        ${RBDL_LIBRARY}
        OsqpEigen::OsqpEigen
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(MPCC_LIB PUBLIC OpenMP::OpenMP_CXX)
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