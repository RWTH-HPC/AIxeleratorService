cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_CXX_STANDARD 17)

project(AIxeleratorService VERSION 0.01 LANGUAGES CXX C Fortran)

# CMP0074: find_package() uses <PackageName>_ROOT variables.
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_BUILD_TYPE ReleaseWithDebInfo)

option(BUILD_SHARED_LIBS "Build shared instead of static libraries" ON)
option(BUILD_TESTS "Build tests" ON)

# Options for AI-Frameworks
option(WITH_TORCH "Build with support for Torch backend" OFF)
option(WITH_TENSORFLOW "Build with support for Tensorflow backend" OFF)

# add error check if no AI framework is activated
if(NOT ${WITH_TORCH} AND NOT ${WITH_TENSORFLOW})
  message(FATAL_ERROR "You are trying to build the AIxeleratorService without any ML framework backend. At least one of the following options need to be defined: \n -DWITH_TORCH=ON for PyTorch libtorch backend \n -DWITH_TENSORFLOW=ON for TensorFlow backend")
endif()

find_package(MPI REQUIRED)

if(WITH_TORCH)
  find_package(Torch REQUIRED)

  # remove libcuda from TORCH_LIBRARIES
  message(STATUS "Torch Libraries defined by FindTorch = ${TORCH_LIBRARIES}")
  set(TORCH_LIBRARIES_TMP "")
  foreach (_lib ${TORCH_LIBRARIES})
    string(FIND "${_lib}" "libcuda.so" LIB_CUDA_CONTAINED)
    if(LIB_CUDA_CONTAINED EQUAL -1)
      if(TORCH_LIBRARIES_TMP STREQUAL "")
        set(TORCH_LIBRARIES_TMP "${_lib}")
      else()
        set(TORCH_LIBRARIES_TMP "${TORCH_LIBRARIES_TMP};${_lib}")
      endif()
    endif()
  endforeach()
  message(STATUS "modified Torch Libraries = ${TORCH_LIBRARIES_TMP}")
  set(TORCH_LIBRARIES ${TORCH_LIBRARIES_TMP})
endif()

if(WITH_TENSORFLOW)
  # simulate a findTensorflow.cmake
  # Tensorflow_DIR needs to be passed to CMAKE configure
  message(STATUS "Tensorflow_DIR = ${Tensorflow_DIR}")
  set(TENSORFLOW_INCLUDE_DIRS ${Tensorflow_DIR}/include)
  message(STATUS "TENSORFLOW_INCLUDE_DIRS = ${TENSORFLOW_INCLUDE_DIRS}")
  set(TENSORFLOW_LIB_DIR ${Tensorflow_DIR}/lib)
  message(STATUS "TENSORFLOW_LIB_DIR = ${TENSORFLOW_LIB_DIR}")
  set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIB_DIR}/libtensorflow.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so)
  message(STATUS "TENSORFLOW_LIBRARIES = ${TENSORFLOW_LIBRARIES}")
endif()

add_subdirectory(src)

if(BUILD_TESTS)
  add_subdirectory(test)
endif()