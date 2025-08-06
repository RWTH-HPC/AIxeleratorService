cmake_minimum_required(VERSION 3.16.3)

project(AIxeleratorService VERSION 1.00 LANGUAGES CXX C Fortran)



set(CMAKE_CXX_STANDARD 17)
add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=1")
add_compile_options("-g")

# all dependent libraries will be installed to ${CMAKE_INSTALL_PREFIX}/bin, so that they can be found via this RPATH setting
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

include(FetchContent)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "Found CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "Found CUDA compiler version: ${CMAKE_CUDA_COMPILER_VERSION}")

    # determine individual MAJOR, MINOR, PATCH version of CUDA
    string(REPLACE "." ";" CUDA_VERSION_LIST "${CMAKE_CUDA_COMPILER_VERSION}")
    list(GET CUDA_VERSION_LIST 0 CUDA_VERSION_MAJOR)
    list(GET CUDA_VERSION_LIST 1 CUDA_VERSION_MINOR)
    list(GET CUDA_VERSION_LIST 2 CUDA_VERSION_PATCH)
else()
    message(STATUS "CUDA language not available on this system.")
endif()

include(FortranCInterface)
FortranCInterface_VERIFY(CXX)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)

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
set(TORCH_VERSION "2.7.1" CACHE STRING "Version of prebuilt Torch to use. Version format: <MAJOR>.<MINOR>.<PATCH>")
option(WITH_TENSORFLOW "Build with support for Tensorflow backend" OFF)
set(TENSORFLOW_VERSION "2.17.0" CACHE STRING "Version of prebuilt TensorFlow to use.")
option(WITH_SOL "Build with support for NEC SOL backend" OFF)
option(WITH_ONNX "Build with support for ONNX backend" OFF)
set(ONNXRUNTIME_VERSION "1.22.0" CACHE STRING "Version of prebuilt ONNXRuntime to use.")
option(WITH_SCOREP "Build with Score-P instrumentation for profiling/tracing" OFF)

if(CMAKE_CUDA_COMPILER)
  if(WITH_TORCH)
    # define architecture for Torch, e.g. "cu126" (CUDA 12.6) or "cpu"
    set(TORCH_ARCH "cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}")
  endif()
  if(WITH_ONNX)
    set(ONNX_ARCH "-gpu")
  endif()
else()
  if(WITH_TORCH)
    set(TORCH_ARCH "cpu") 
  endif()
  if(WITH_ONNX)
    set(ONNX_ARCH "") # onnx does not explicitly name their libraries "cpu" but omits the "gpu" term
  endif()
endif()

# add error check if no AI framework is activated
if((NOT ${WITH_TORCH} AND NOT ${WITH_TENSORFLOW}) AND NOT ${WITH_SOL} AND NOT ${WITH_ONNX})
  message(FATAL_ERROR "You are trying to build the AIxeleratorService without any ML framework backend. At least one of the following options need to be defined: \n -DWITH_TORCH=ON for PyTorch libtorch backend \n -DWITH_TENSORFLOW=ON for TensorFlow backend \n -DWITH_ONNX=ON for ONNX runtime backend")
endif()

# general packages
find_package(MPI REQUIRED)
if(WITH_SCOREP)
  find_package(Scorep REQUIRED)
endif()

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

  # install numpy
  execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/pip install numpy
      RESULT_VARIABLE NUMPY_RESULT
      COMMAND_ECHO STDOUT
  )
  if(NOT NUMPY_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to pip install numpy! Error code: ${NUMPY_RESULT}")
  endif()

  # install pytorch 
  if(WITH_TORCH OR WITH_ONNX) # (also required to create a test model for ONNXRuntime in Python)
    execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/pip install torch==${TORCH_VERSION}
      RESULT_VARIABLE TORCH_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT TORCH_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to pip install Torch ${TORCH_VERSION}! Error code: ${TORCH_RESULT}")
    endif()
  endif()

  if(WITH_TENSORFLOW)
    execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/pip install tensorflow==${TENSORFLOW_VERSION}
      RESULT_VARIABLE TF_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT TF_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to pip install TensorFlow ${TENSORFLOW_VERSION}! Error code: ${TF_RESULT}")
    endif()

    execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/pip install tf-keras==${TENSORFLOW_VERSION}
      RESULT_VARIABLE TF_KERAS_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT TF_KERAS_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to pip install tf-keras ${TENSORFLOW_VERSION}! Error code: ${TF_KERAS_RESULT}")
    endif()
  endif()

  if(WITH_ONNX)
    # pip install onnx
    execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/pip install onnx
      RESULT_VARIABLE ONNX_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT ONNX_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to pip install onnx ${ONNX_VERSION}! Error code: ${ONNX_RESULT}")
    endif()

    # pip install onnxruntime
    execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/pip install onnxruntime${ONNX_ARCH}==${ONNXRUNTIME_VERSION}
      RESULT_VARIABLE ONNX_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT ONNX_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to pip install ONNXRuntime ${ONNX_VERSION}! Error code: ${ONNX_RESULT}")
    endif()   
  endif()

else()
  message(STATUS "Found existing Python virtual environment: ${AIX_VENV_DIR}")
endif()
###################### END CREATION OF VIRTUAL ENVIRONMENT ######################

if(WITH_TORCH)
  add_compile_options("-DWITH_TORCH")

  set(TORCH_SOURCE_DIR "${CMAKE_BINARY_DIR}/extern/torch")
  message(STATUS "TORCH_SOURCE_DIR = ${TORCH_SOURCE_DIR}")

  # TODO: download libtorch CPU or GPU
  set(TORCH_DOWNLOAD_URL "https://download.pytorch.org/libtorch/${TORCH_ARCH}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${TORCH_ARCH}.zip")
  FetchContent_Declare(
    TorchDownload
    URL ${TORCH_DOWNLOAD_URL}
    SOURCE_DIR "${TORCH_SOURCE_DIR}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(TorchDownload)

  list(APPEND CMAKE_PREFIX_PATH ${TORCH_SOURCE_DIR})
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

  # add torch libraries required for installation to be found via RPATH
  set(TORCH_LIB_DIR "${TORCH_SOURCE_DIR}/lib")
  message(STATUS "TORCH_LIB_DIR = ${TORCH_LIB_DIR}")
  file(GLOB TORCH_CORE_LIBS "${TORCH_LIB_DIR}/libtorch*.so")
  message(STATUS "TORCH_CORE_LIBS = ${TORCH_CORE_LIBS}")
  file(GLOB TORCH_C10_LIBS "${TORCH_LIB_DIR}/libc10*.so")
  message(STATUS "TORCH_C10_LIBS = ${TORCH_C10_LIBS}")
  set(TORCH_INSTALL_LIBS ${TORCH_CORE_LIBS} ${TORCH_C10_LIBS})
  message(STATUS "TORCH_INSTALL_LIBS = ${TORCH_INSTALL_LIBS}")

  file(GLOB TORCH_INSTALL_LIBS "${TORCH_LIB_DIR}/*.so*")
  install(
    FILES ${TORCH_INSTALL_LIBS}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  )
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
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(TensorFlowDownload)

  set(TENSORFLOW_PROTOBUF_INCLUDE "${AIX_VENV_DIR}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/tensorflow/include/")

  set(TENSORFLOW_INCLUDE_DIRS ${Tensorflow_DIR}/include ${TENSORFLOW_PROTOBUF_INCLUDE})
  message(STATUS "TENSORFLOW_INCLUDE_DIRS = ${TENSORFLOW_INCLUDE_DIRS}")
  set(TENSORFLOW_LIB_DIR ${Tensorflow_DIR}/lib)
  message(STATUS "TENSORFLOW_LIB_DIR = ${TENSORFLOW_LIB_DIR}")
  set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIB_DIR}/libtensorflow.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so)
  message(STATUS "TENSORFLOW_LIBRARIES = ${TENSORFLOW_LIBRARIES}")

  # modern CMake solution using IMPORTED targets
  add_library(TensorFlow::tensorflow SHARED IMPORTED GLOBAL)
  set_target_properties(TensorFlow::tensorflow PROPERTIES
    IMPORTED_LOCATION "${TENSORFLOW_LIB_DIR}/libtensorflow.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TENSORFLOW_INCLUDE_DIRS}"
    IMPORTED_SONAME "libtensorflow.so"
    INTERFACE_LINK_DIRECTORIES ${TENSORFLOW_LIB_DIR}
  )

  # modern CMake solution using IMPORTED targets
  add_library(TensorFlow::framework SHARED IMPORTED GLOBAL)
  set_target_properties(TensorFlow::framework PROPERTIES
    IMPORTED_LOCATION "${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so"
    IMPORTED_SONAME "libtensorflow_framework.so"
    INTERFACE_LINK_DIRECTORIES "${TENSORFLOW_LIB_DIR}"
  )

  file(GLOB TENSORFLOW_CORE_LIBS "${TENSORFLOW_LIB_DIR}/libtensorflow*.so*")
  file(GLOB TENSORFLOW_FRAMEWORK_LIBS "${TENSORFLOW_LIB_DIR}/libtensorflow_framework*.so*")
  set(TENSORFLOW_SHARED_LIBS ${TENSORFLOW_CORE_LIBS} ${TENSORFLOW_FRAMEWORK_LIBS})

  install(
    FILES
      ${TENSORFLOW_SHARED_LIBS}
    DESTINATION
      ${CMAKE_INSTALL_PREFIX}/lib
  )
endif()

if(WITH_ONNX)
  add_compile_options("-DWITH_ONNX")

  # URL 
  set(ONNX_DIR "${CMAKE_BINARY_DIR}/extern/onnxruntime")
  message(STATUS "ONNX_DIR = ${ONNX_DIR}")

  # download TensorFlow
  FetchContent_Declare(
    ONNXRuntimeDownload
    URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64${ONNX_ARCH}-1.22.0.tgz"
    SOURCE_DIR "${ONNX_DIR}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(ONNXRuntimeDownload)

  message(STATUS "ONNX_DIR = ${ONNX_DIR}")
  set(ONNX_INCLUDE_DIRS ${ONNX_DIR}/include)
  message(STATUS "ONNX_INCLUDE_DIRS = ${ONNX_INCLUDE_DIRS}")
  set(ONNX_LIB_DIR ${ONNX_DIR}/lib)
  message(STATUS "ONNX_LIB_DIR = ${ONNX_LIB_DIR}")
  set(ONNX_LIBRARIES ${ONNX_LIB_DIR}/libonnxruntime.so)
  message(STATUS "ONNX_LIBRARIES = ${ONNX_LIBRARIES}")

  # modern CMake solution using IMPORTED targets
  add_library(ONNXRuntime::onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(ONNXRuntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNX_LIB_DIR}/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNX_INCLUDE_DIRS}"
    IMPORTED_SONAME "libonnxruntime.so"
    INTERFACE_LINK_DIRECTORIES ${ONNX_LIB_DIR}
  )

  file(GLOB ONNX_INSTALL_LIBS "${ONNX_LIB_DIR}/libonnxruntime*.so*")
  install(
    FILES
      ${ONNX_INSTALL_LIBS}
    DESTINATION
      ${CMAKE_INSTALL_PREFIX}/lib
  )
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
add_subdirectory(models)


if(BUILD_TESTS)
  add_subdirectory(test)
endif()

