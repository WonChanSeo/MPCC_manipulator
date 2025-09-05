include(FetchContent)

message(STATUS "Fetching/configuring QDLDL solver")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

# FetchContent_Declare(
#   qdldl
#   GIT_REPOSITORY https://github.com/osqp/qdldl.git
#   GIT_TAG v0.1.8
#   )

FetchContent_Declare(
  qdldl
  SOURCE_DIR "/home/mms-wonchan/osqp/qdldl"  # ← 여기를 로컬 클론 위치로 바꿔주세요
)

# Make QDLDL use the same types as OSQP
set(QDLDL_FLOAT ${OSQP_USE_FLOAT} CACHE BOOL "QDLDL Float type")
set(QDLDL_LONG ${OSQP_USE_LONG} CACHE BOOL "QDLDL Integer type")

# We only want the object library, so turn off the other library products
set(QDLDL_BUILD_STATIC_LIB OFF CACHE BOOL "Build QDLDL static library")
set(QDLDL_BUILD_SHARED_LIB OFF CACHE BOOL "Build QDLDL shared library")

FetchContent_MakeAvailable(qdldl)
FetchContent_GetProperties(qdldl)


# ────────────────────────────────────────────────────────────────
# QDLDL 객체 라이브러리에 half-precision 옵션 적용
if(OSQP_USE_HALF)
  # 1) 코드 내 분기 활성화
  target_compile_definitions(qdldlobject PUBLIC OSQP_USE_HALF)

  # 2) C2x 모드로 _Float16 활성화
  set_target_properties(qdldlobject PROPERTIES
    C_STANDARD           23
    C_STANDARD_REQUIRED  ON
    C_EXTENSIONS         NO
  )

  # 3) 아키텍처별 하드웨어 half 명령어 활성화
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_compile_options(qdldlobject PRIVATE -std=c2x -march=armv8.2-a+fp16)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    target_compile_options(qdldlobject PRIVATE -std=c2x -mf16c)
  else()
    target_compile_options(qdldlobject PRIVATE -std=c2x)
  endif()
endif()
# ────────────────────────────────────────────────────────────────

list(POP_BACK CMAKE_MESSAGE_INDENT)

set_source_files_properties($<TARGET_OBJECTS:qdldlobject> PROPERTIES GENERATED 1)

file(
    GLOB
    AMD_SRC_FILES
    CONFIGURE_DEPENDS
    ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/amd/src/*.c
    ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/amd/include/*.h
    )

set( LIN_SYS_QDLDL_NON_EMBEDDED_SRC_FILES
     ${AMD_SRC_FILES}
     )

set( LIN_SYS_QDLDL_EMBEDDED_SRC_FILES
     ${OSQP_ALGEBRA_ROOT}/_common/kkt.h
     ${OSQP_ALGEBRA_ROOT}/_common/kkt.c
     ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/qdldl_interface.h
     ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/qdldl_interface.c
     )

set( LIN_SYS_QDLDL_SRC_FILES
     ${LIN_SYS_QDLDL_EMBEDDED_SRC_FILES}
     ${LIN_SYS_QDLDL_NON_EMBEDDED_SRC_FILES}
     )

set( LIN_SYS_QDLDL_INC_PATHS
     ${qdldl_include}
     ${OSQP_ALGEBRA_ROOT}/_common/
     ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/
     ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/amd/include
     ${qdldl_SOURCE_DIR}/include
     ${qdldl_BINARY_DIR}/include
     )
