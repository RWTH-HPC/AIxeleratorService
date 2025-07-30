cmake_minimum_required(VERSION 3.16.3)

project(AIxeleratorService VERSION 1.00 LANGUAGES CXX C Fortran)

set(CMAKE_CXX_STANDARD 17)
add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=1")
add_compile_options("-g")

include(FetchContent)

include(FortranCInterface)
FortranCInterface_VERIFY(CXX)

# CMP0074: find_package() uses <PackageName>_ROOT variables.
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_BUILD_TYPE Release)
#add_compile_options("-fsanitize=address")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(BUILD_SHARED_LIBS "Build shared instead of static libraries" ON)
option(BUILD_TESTS "Build tests" ON)

# Options for AI-Frameworks
option(WITH_TORCH "Build with support for Torch backend" OFF)
option(WITH_TENSORFLOW "Build with support for Tensorflow backend" OFF)
set(TENSORFLOW_VERSION "2.17.0" CACHE STRING "Version of TensorFlow to use.")
option(WITH_SOL "Build with support for NEC SOL backend" OFF)
option(WITH_ONNX "Build with support for ONNX backend" OFF)
option(WITH_SCOREP "Build with Score-P instrumentation for profiling/tracing" OFF)

# creation of virtual environment
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(AIX_VENV_DIR "${CMAKE_BINARY_DIR}/extern/venv-py${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}${Python3_VERSION_PATCH}")
# check if the environment already exists
if(NOT EXISTS "${AIX_VENV_DIR}/bin/activate")
  message(STATUS "No existing Python virtual environment found! Installing now!")
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m venv ${AIX_VENV_DIR}
    RESULT_VARIABLE VENV_RESULT
    COMMAND_ECHO STDOUT
  )
  if(NOT VENV_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to create Python virtual environment! Error code: ${VENV_RESULT}")
  endif()

  # install required packages
  if(WITH_TENSORFLOW)
    execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/pip install tensorflow==${TENSORFLOW_VERSION}
      RESULT_VARIABLE TF_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT TF_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to pip install TensorFlow! Error code: ${TF_RESULT}")

    endif()

    execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/pip install tf-keras==${TENSORFLOW_VERSION}
      RESULT_VARIABLE TF_KERAS_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT TF_KERAS_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to pip install tf-keras! Error code: ${TF_KERAS_RESULT}")
    endif()
  endif()

else()
  message(STATUS "Found existing Python virtual environment: ${AIX_VENV_DIR}")
endif()
###################### END CREATION OF VIRTUAL ENVIRONMENT ######################

# add error check if no AI framework is activated
if((NOT ${WITH_TORCH} AND NOT ${WITH_TENSORFLOW}) AND NOT ${WITH_SOL} AND NOT ${WITH_ONNX})
  message(FATAL_ERROR "You are trying to build the AIxeleratorService without any ML framework backend. At least one of the following options need to be defined: \n -DWITH_TORCH=ON for PyTorch libtorch backend \n -DWITH_TENSORFLOW=ON for TensorFlow backend \n -DWITH_ONNX=ON for ONNX runtime backend")
endif()

find_package(MPI REQUIRED)
if(WITH_SCOREP)
  find_package(Scorep REQUIRED)
endif()

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

  set(Tensorflow_DIR "${CMAKE_BINARY_DIR}/extern/tensorflow")
  message(STATUS "Tensorflow_DIR = ${Tensorflow_DIR}")

  # download TensorFlow
  FetchContent_Declare(
    TensorFlowDownload
    URL "https://storage.googleapis.com/tensorflow/versions/${TENSORFLOW_VERSION}/libtensorflow-gpu-linux-x86_64.tar.gz"
    SOURCE_DIR "${Tensorflow_DIR}"
  )
  FetchContent_MakeAvailable(TensorFlowDownload)

  set(TENSORFLOW_PROTOBUF_INCLUDE "${AIX_VENV_DIR}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/tensorflow/include/")

  set(TENSORFLOW_INCLUDE_DIRS ${Tensorflow_DIR}/include ${TENSORFLOW_PROTOBUF_INCLUDE})
  message(STATUS "TENSORFLOW_INCLUDE_DIRS = ${TENSORFLOW_INCLUDE_DIRS}")
  set(TENSORFLOW_LIB_DIR ${Tensorflow_DIR}/lib)
  message(STATUS "TENSORFLOW_LIB_DIR = ${TENSORFLOW_LIB_DIR}")
  set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIB_DIR}/libtensorflow.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so)
  message(STATUS "TENSORFLOW_LIBRARIES = ${TENSORFLOW_LIBRARIES}")
endif()

if(WITH_ONNX)
  add_compile_options("-DWITH_ONNX")
  message(STATUS "ONNX_DIR = ${ONNX_DIR}")
  set(ONNX_INCLUDE_DIRS ${ONNX_DIR}/include)
  message(STATUS "ONNX_INCLUDE_DIRS = ${ONNX_INCLUDE_DIRS}")
  set(ONNX_LIB_DIR ${ONNX_DIR}/lib)
  message(STATUS "ONNX_LIB_DIR = ${ONNX_LIB_DIR}")
  set(ONNX_LIBRARIES ${ONNX_LIB_DIR}/libonnxruntime.so)
  message(STATUS "ONNX_LIBRARIES = ${ONNX_LIBRARIES}")
endif()

if(WITH_SOL)
  add_compile_options("-DWITH_SOL")
  message(STATUS "VEDA_DIR = ${VEDA_DIR}") 
  set(VEDA_INCLUDE_DIRS ${VEDA_DIR}/include)
  message(STATUS "VEDA_INCLUDE_DIRS = ${VEDA_INCLUDE_DIRS}")
  set(VEDA_LIBRARIES ${VEDA_DIR}/lib64/libveda.so.0)
  message(STATUS "VEDA_LIBRARIES = ${VEDA_LIBRARIES}")
endif()

if(WITH_SCOREP)
  add_compile_options("-DSCOREP")
  message(STATUS "AIxeleratorService will be build with Score-P intrumentation!")
endif()

add_subdirectory(src)

if(BUILD_TESTS)
  add_subdirectory(test)
endif()

