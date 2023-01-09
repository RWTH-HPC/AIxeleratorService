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

if(BUILD_TESTS)
  add_subdirectory(test)
endif()