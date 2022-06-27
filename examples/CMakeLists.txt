project(aixeleratorServiceMWE)
cmake_minimum_required(VERSION 3.21)
enable_language(CXX)
find_package(MPI REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_BUILD_TYPE DEBUG)
message(STATUS "MPI include: ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI libraries: ${MPI_CXX_LIBRARIES}")
include_directories(${CMAKE_CURRENT_LIST_DIR}/../aixeleratorService ${MPI_CXX_INCLUDE_DIRS})
set(SOURCES
    example-mwe.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../aixeleratorService/aixeleratorService.cpp
    )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=broadwell")
add_executable(example-mwe.x ${SOURCES})
target_link_libraries(example-mwe.x ${MPI_CXX_LIBRARIES} ${TORCH_LIBRARIES})
set_property(TARGET example-mwe.x PROPERTY CXX_STANDARD 14)


