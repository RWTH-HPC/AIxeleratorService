cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_CXX_STANDARD 17)

project(AIxeleratorService VERSION 0.01 LANGUAGES CXX)

# CMP0074: find_package() uses <PackageName>_ROOT variables.
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_BUILD_TYPE Debug)

option(BUILD_SHARED_LIBS "Build shared instead of static libraries" ON)
option(BUILD_TESTS "Build tests" ON)

find_package(Torch REQUIRED)
find_package(MPI REQUIRED)

# uncomment to inspect all variables in CMAKE
#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

# remove libcuda from TORCH_LIBRARIES
message(STATUS "Torch Libraries defined by FindTorch = ${TORCH_LIBRARIES}")
set(TORCH_LIBRARIES_TMP "")
foreach (_lib ${TORCH_LIBRARIES})
  #message(STATUS "Current TORCH_LIBRARIES_TMP = ${TORCH_LIBRARIES_TMP}")
  string(FIND "${_lib}" "libcuda.so" LIB_CUDA_CONTAINED)
  #message(STATUS "Current lib = ${_lib}")
  if(LIB_CUDA_CONTAINED EQUAL -1)
    #message(STATUS "Current lib = ${_lib} is not CUDA")
    if(TORCH_LIBRARIES_TMP STREQUAL "")
      #message(STATUS "Current lib = ${_lib} equal empty")
      set(TORCH_LIBRARIES_TMP "${_lib}")
    else()
      #message(STATUS "Current lib = ${_lib} notequal empty")
      set(TORCH_LIBRARIES_TMP "${TORCH_LIBRARIES_TMP};${_lib}")
    endif()
  endif()
endforeach()
message(STATUS "modified Torch Libraries = ${TORCH_LIBRARIES_TMP}")
set(TORCH_LIBRARIES ${TORCH_LIBRARIES_TMP})

# simulate a findTensorflow.cmake
# Tensorflow_DIR needs to be passed to CMAKE configure
set(TENSORFLOW_INCLUDE_DIRS ${Tensorflow_DIR}/include)
set(TENSORFLOW_LIB_DIR ${Tensorflow_DIR}/lib)
set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIB_DIR}/libtensorflow.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so)

# find OpenFOAM
set(OPENFOAM_INCLUDE_DIRS ${OPENFOAM_DIR}/src/OpenFOAM/lnInclude ${OPENFOAM_DIR}/src/finiteVolume/lnInclude)

add_subdirectory(src)

if(BUILD_TESTS)
  add_subdirectory(test)
endif()