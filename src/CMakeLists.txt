# Add AIxelerator as an empty target
add_library(AIxelerator)
set_target_properties(
  AIxelerator
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED Yes
             CXX_EXTENSIONS No
             VERSION ${AIxelerator_VERSION}
             SOVERSION ${AIxelerator_SOVERSION})

# Add source files to target
target_sources(
  AIxelerator PRIVATE core/data_distributor.cpp models/TorchModel.cpp
                      models/AIModel.cpp models/AIModelUtils.cpp)

target_include_directories(
  AIxelerator
  PUBLIC ${CMAKE_SOURCE_DIR}/src/core ${CMAKE_SOURCE_DIR}/src/frameworks
         ${CMAKE_SOURCE_DIR}/src/models)

# Setup MPI
target_link_libraries(AIxelerator PRIVATE MPI::MPI_CXX)
# Setup Torch
target_link_libraries(AIxelerator PUBLIC ${TORCH_LIBRARIES})
target_include_directories(AIxelerator PUBLIC ${TORCH_INCLUDE_DIRS})
