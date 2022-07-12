# Add AIxelerator as an empty target
add_library(AIxelerator)
set_target_properties(AIxelerator PROPERTIES
  # precice is a C++14 project
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED Yes
  CXX_EXTENSIONS No
  VERSION ${AIxelerator_VERSION}
  SOVERSION ${AIxelerator_SOVERSION}
  )

# Add source files to target
target_sources(AIxelerator PRIVATE
    aixeleratorService.cpp
    frameworks/pytorch/torchInference.cpp 
    models/TorchModel.cpp
)

target_include_directories(AIxelerator PRIVATE
    src/frameworks
    src/models
    )

