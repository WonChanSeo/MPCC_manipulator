# QDLDL Precision Test
cmake_minimum_required(VERSION 3.10)
project(qdldl_precision_test C)

# Read QDLDL configuration from parent CMake cache
# This is set by the run_precision_tests.sh script
option(QDLDL_USE_FLEXFLOAT "Use FlexFloat for custom precision" OFF)
set(FF_EXPONENT_BITS "8" CACHE STRING "FlexFloat exponent bits")
set(FF_MANTISSA_BITS "23" CACHE STRING "FlexFloat mantissa bits")

# Find QDLDL library
find_library(QDLDL_LIBRARY NAMES qdldl PATHS ${CMAKE_SOURCE_DIR}/../lib/lib NO_DEFAULT_PATH)
find_path(QDLDL_INCLUDE_DIR qdldl.h PATHS ${CMAKE_SOURCE_DIR}/../External/qdldl/include NO_DEFAULT_PATH)
find_path(QDLDL_BUILD_INCLUDE_DIR qdldl_types.h PATHS ${CMAKE_SOURCE_DIR}/../lib/include/qdldl NO_DEFAULT_PATH)
find_path(FLEXFLOAT_DIR flexfloat.h PATHS ${CMAKE_SOURCE_DIR}/../External/flexfloat/include NO_DEFAULT_PATH)

if(NOT QDLDL_LIBRARY OR NOT QDLDL_INCLUDE_DIR OR NOT QDLDL_BUILD_INCLUDE_DIR)
    message(FATAL_ERROR "QDLDL library not found. Please build OSQP first.")
endif()

message(STATUS "QDLDL library: ${QDLDL_LIBRARY}")
message(STATUS "QDLDL include: ${QDLDL_INCLUDE_DIR}")
message(STATUS "QDLDL build include: ${QDLDL_BUILD_INCLUDE_DIR}")

# Create executable
add_executable(qdldl_precision_test qdldl_precision_test.c)

target_include_directories(qdldl_precision_test PRIVATE
    ${QDLDL_INCLUDE_DIR}
    ${QDLDL_BUILD_INCLUDE_DIR}
)

# Add FlexFloat support if enabled
if(QDLDL_USE_FLEXFLOAT)
    if(FLEXFLOAT_DIR)
        message(STATUS "FlexFloat enabled: E${FF_EXPONENT_BITS}M${FF_MANTISSA_BITS}")
        target_include_directories(qdldl_precision_test PRIVATE ${FLEXFLOAT_DIR})
        target_compile_definitions(qdldl_precision_test PRIVATE
            QDLDL_USE_FLEXFLOAT
            FF_exponent_bits=${FF_EXPONENT_BITS}
            FF_mantissa_bits=${FF_MANTISSA_BITS}
        )
    else()
        message(WARNING "FlexFloat directory not found, disabling FlexFloat")
    endif()
else()
    message(STATUS "FlexFloat disabled (using standard double precision)")
endif()

target_link_libraries(qdldl_precision_test ${QDLDL_LIBRARY})

# Link math library
target_link_libraries(qdldl_precision_test m)

# Install
install(TARGETS qdldl_precision_test DESTINATION ${CMAKE_SOURCE_DIR}/../bin)
