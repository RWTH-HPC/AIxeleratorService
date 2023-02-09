add_library(AIxeleratorServiceLib SHARED)
target_sources(
    AIxeleratorServiceLib PRIVATE 
        distributionStrategy/roundRobinDistribution.cpp inferenceStrategy/torchInference.cpp inferenceStrategy/tensorflowInference.cpp aixeleratorService.cpp
)

target_include_directories(
    AIxeleratorServiceLib
    PUBLIC  ${CMAKE_SOURCE_DIR}/src/distributionStrategy
            ${CMAKE_SOURCE_DIR}/src/inferenceStrategy
            ${CMAKE_SOURCE_DIR}/src/utils
            ${MPI_INCLUDE_PATH}
            ${TORCH_INCLUDE_DIRS}
            ${TENSORFLOW_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR}/src/
)

target_link_libraries(
    AIxeleratorServiceLib 
    PRIVATE ${MPI_LIBRARIES}
)

target_link_libraries(
    AIxeleratorServiceLib 
    PUBLIC  ${TORCH_LIBRARIES}
            ${TENSORFLOW_LIBRARIES} 
)