cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_CXX_STANDARD 17)

project(AIxeleratorService VERSION 0.01 LANGUAGES CXX)

# CMP0074: find_package() uses <PackageName>_ROOT variables.
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_BUILD_TYPE Debug)

option(BUILD_TESTS "Build tests" ON)

find_package(Torch REQUIRED)
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()

# simulate a findTensorflow.cmake
# Tensorflow_DIR needs to be passed to CMAKE configure
set(TENSORFLOW_INCLUDE_DIRS ${Tensorflow_DIR}/include)
set(TENSORFLOW_LIB_DIR ${Tensorflow_DIR}/lib)
set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIB_DIR}/libtensorflow.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so)

if(BUILD_TESTS)
  add_subdirectory(test)
endif()