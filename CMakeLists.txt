cmake_minimum_required (VERSION 3.16.3)

set(CMAKE_CXX_STANDARD 17)

# Add cmake convenience tool project
# For more information see:
# https://github.com/aminya/project_options/
include(FetchContent)
FetchContent_Declare(_project_options URL https://github.com/aminya/project_options/archive/refs/tags/v0.22.4.zip)
FetchContent_MakeAvailable(_project_options)
include(${_project_options_SOURCE_DIR}/Index.cmake)

project(AIxelerator VERSION 0.01 LANGUAGES CXX)
set(AIxelerator_SOVERSION ${AIxelerator_VERSION_MAJOR})

#
# Overview of this configuration
#
# PREAMBLE
# Setup Options
# Find Mandatory Dependencies
# Find Configurable Dependencies
# Configuration of Target AIxelerator
# Install Targets for AIxelerator
# CPack
#

#
# PREAMBLE
#

# Make our cmake modules visible to CMake
list (APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
list (APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

# CMake Policies

# CMP0074: find_package() uses <PackageName>_ROOT variables.
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

#
# Setup Options
#

if(NOT CMAKE_BUILD_TYPE)
  message(WARNING "You did not specify a CMAKE_BUILD_TYPE.
  We will assume you asked for a Release build.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "The type of this build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

option(BUILD_SHARED_LIBS "Build shared instead of static libraries" ON)
option(BUILD_TESTING "Build tests" OFF)
option(AIxelerator_Packages "Configure package generation." OFF)

# Print information about this configuration
include(PrintTargetProperties)
include(PrintHelper)
print_configuration(
  ADDITIONAL
  "AIxelerator_Packages;Configure package generation"
  )
print_empty()

#
# Find Mandatory Dependencies
#

find_package(MPI REQUIRED)
find_package(Torch REQUIRED)

#
# Setup miscellaneous features
#

print_empty()
print_section("TARGETS & PACKAGES")

# # Add AIxelerator as an empty target
# add_library(AIxelerator)
# set_target_properties(AIxelerator PROPERTIES
#   # precice is a C++14 project
#   CXX_STANDARD 14
#   CXX_STANDARD_REQUIRED Yes
#   CXX_EXTENSIONS No
#   VERSION ${AIxelerator_VERSION}
#   SOVERSION ${AIxelerator_SOVERSION}
#   )
add_subdirectory(src)

# add_subdirectory(examples)

add_subdirectory(test)
