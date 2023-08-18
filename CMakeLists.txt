cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_CXX_STANDARD 17)

project(AIxeleratorService VERSION 0.01 LANGUAGES CXX C Fortran)

include(FortranCInterface)
FortranCInterface_VERIFY(CXX)

# CMP0074: find_package() uses <PackageName>_ROOT variables.
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_BUILD_TYPE Release)

option(BUILD_SHARED_LIBS "Build shared instead of static libraries" ON)
option(BUILD_TESTS "Build tests" ON)

# Options for AI-Frameworks
option(WITH_TORCH "Build with support for Torch backend" OFF)
option(WITH_TENSORFLOW "Build with support for Tensorflow backend" OFF)
option(WITH_SOL "Build with support for NEC SOL backend" OFF)

# add error check if no AI framework is activated
if((NOT ${WITH_TORCH} AND NOT ${WITH_TENSORFLOW}) AND NOT ${WITH_SOL})
  message(FATAL_ERROR "You are trying to build the AIxeleratorService without any ML framework backend. At least one of the following options need to be defined: \n -DWITH_TORCH=ON for PyTorch libtorch backend \n -DWITH_TENSORFLOW=ON for TensorFlow backend")
endif()

find_package(MPI REQUIRED)

add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")

if(WITH_TORCH)
  add_compile_options("-DWITH_TORCH")
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
  add_compile_options("-DWITH_TENSORFLOW")
  # only needed for newer versions of TF (e.g. 2.11 but maybe also lower)
  # simulate a findTensorflow.cmake
  # Tensorflow_DIR needs to be passed to CMAKE configure
  message(STATUS "Tensorflow_DIR = ${Tensorflow_DIR}")
  set(TENSORFLOW_PROTOBUF_INCLUDE ${Tensorflow_Python_DIR}/include/)
  set(TENSORFLOW_INCLUDE_DIRS ${Tensorflow_DIR}/include ${TENSORFLOW_PROTOBUF_INCLUDE})
  message(STATUS "TENSORFLOW_INCLUDE_DIRS = ${TENSORFLOW_INCLUDE_DIRS}")
  set(TENSORFLOW_LIB_DIR ${Tensorflow_DIR}/lib)
  message(STATUS "TENSORFLOW_LIB_DIR = ${TENSORFLOW_LIB_DIR}")
  set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIB_DIR}/libtensorflow.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so)
  message(STATUS "TENSORFLOW_LIBRARIES = ${TENSORFLOW_LIBRARIES}")
endif()

if(WITH_SOL)
  add_compile_options("-DWITH_SOL")
  message(STATUS "VEDA_DIR = ${VEDA_DIR}") 
  set(VEDA_INCLUDE_DIRS ${VEDA_DIR}/include)
  message(STATUS "VEDA_INCLUDE_DIRS = ${VEDA_INCLUDE_DIRS}")
  set(VEDA_LIBRARIES ${VEDA_DIR}/lib64/libveda.so.0)
  message(STATUS "VEDA_LIBRARIES = ${VEDA_LIBRARIES}")
endif()

add_subdirectory(src)

if(BUILD_TESTS)
  add_subdirectory(test)
endif()

