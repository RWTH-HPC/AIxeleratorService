add_library(AIxeleratorServiceLib SHARED)
set_target_properties(AIxeleratorServiceLib PROPERTIES OUTPUT_NAME AIxeleratorService)
target_sources(
    AIxeleratorServiceLib 
        PRIVATE 
            distributionStrategy/roundRobinDistribution.cpp  
            communicationStrategy/collectiveCommunication.cpp
            communicationStrategy/nonBlockingPtoPCommunication.cpp
            aixeleratorService/aixeleratorService.cpp
            aixeleratorService/aixeleratorService_interfaceC.cpp
        PUBLIC
            aixeleratorService/aixeleratorService_interfaceF.f90
)

target_include_directories(
    AIxeleratorServiceLib
    PUBLIC  ${PROJECT_SOURCE_DIR}/src/distributionStrategy
            ${PROJECT_SOURCE_DIR}/src/communicationStrategy
            ${PROJECT_SOURCE_DIR}/src/inferenceStrategy
            ${PROJECT_SOURCE_DIR}/src/utils
            ${MPI_INCLUDE_PATH}     
            ${PROJECT_SOURCE_DIR}/src/
)

target_link_libraries(
    AIxeleratorServiceLib 
    PUBLIC ${MPI_LIBRARIES}
)

if(WITH_SCOREP)
    scorep_instrument_target(
        AIxeleratorServiceLib
            USER ON
            COMPILER OFF
    )
endif()

if(WITH_TORCH)
    target_sources(AIxeleratorServiceLib PRIVATE inferenceStrategy/torchInference/torchInference.cpp)
    target_include_directories(AIxeleratorServiceLib PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(AIxeleratorServiceLib PUBLIC ${TORCH_LIBRARIES})
endif()

if(WITH_TENSORFLOW)
    target_sources(AIxeleratorServiceLib PRIVATE inferenceStrategy/tensorflowInference/tensorflowInference.cpp)
    #target_sources(AIxeleratorServiceLib PRIVATE inferenceStrategy/tensorflowInference/tensorflowInferenceCpp.cpp)

    # target_include_directories(AIxeleratorServiceLib PUBLIC ${TENSORFLOW_INCLUDE_DIRS}) # not necessary anymore when using imported targets
    target_link_libraries(AIxeleratorServiceLib PUBLIC TensorFlow::tensorflow TensorFlow::framework)
endif()

if(WITH_ONNX)
    target_sources(AIxeleratorServiceLib PRIVATE inferenceStrategy/onnxInference/onnxInference.cpp)
    target_include_directories(AIxeleratorServiceLib PUBLIC ${ONNX_INCLUDE_DIRS})
    target_link_libraries(AIxeleratorServiceLib PUBLIC ${ONNX_LIBRARIES})
endif()

if(WITH_SOL)
    target_sources(AIxeleratorServiceLib PRIVATE inferenceStrategy/solInference/solInference.cpp)
    target_include_directories(AIxeleratorServiceLib PUBLIC ${VEDA_INCLUDE_DIRS})
    target_link_libraries(AIxeleratorServiceLib PUBLIC ${VEDA_LIBRARIES})
endif()

set_target_properties(
    AIxeleratorServiceLib 
    PROPERTIES 
    #    PUBLIC_HEADER 
    #        "aixeleratorService/aixeleratorService.h;aixeleratorService/#aixeleratorService_interfaceC.h;aixeleratorService/#aixeleratorService_interfaceF.f90"
        LINKER_LANGUAGE Fortran
)

# install headers
install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/src/"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
    FILES_MATCHING 
        PATTERN "*.h"
)

# TODO: check if required?
# install(
#     FILES ${PROJECT_SOURCE_DIR}/src/aixeleratorService/aixeleratorService_interfaceF.f90
#     DESTINATION "${CMAKE_BINARY_DIR}/include/aixeleratorService"
# )

# install Fortran module
install(
    FILES ${CMAKE_Fortran_MODULE_DIRECTORY}/aixelerator_service_mod.mod
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/aixeleratorService
)

# install shared library
install(
    TARGETS AIxeleratorServiceLib
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    #PUBLIC_HEADER DESTINATION ${CMAKE_BINARY_DIR}/include
)

