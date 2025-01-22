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
    PUBLIC  ${CMAKE_SOURCE_DIR}/src/distributionStrategy
            ${CMAKE_SOURCE_DIR}/src/communicationStrategy
            ${CMAKE_SOURCE_DIR}/src/inferenceStrategy
            ${CMAKE_SOURCE_DIR}/src/utils
            ${MPI_INCLUDE_PATH}     
            ${CMAKE_SOURCE_DIR}/src/
)

target_link_libraries(
    AIxeleratorServiceLib 
    PRIVATE ${MPI_LIBRARIES}
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
    target_include_directories(AIxeleratorServiceLib PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(AIxeleratorServiceLib PUBLIC ${TORCH_LIBRARIES})
endif()

if(WITH_TENSORFLOW)
    target_sources(AIxeleratorServiceLib PRIVATE inferenceStrategy/tensorflowInference/tensorflowInference.cpp)
    #target_sources(AIxeleratorServiceLib PRIVATE inferenceStrategy/tensorflowInference/tensorflowInferenceCpp.cpp)

    target_include_directories(AIxeleratorServiceLib PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
    target_link_libraries(AIxeleratorServiceLib PUBLIC ${TENSORFLOW_LIBRARIES})
endif()

if(WITH_ONNX)
    target_sources(AIxeleratorServiceLib PRIVATE inferenceStrategy/onnxInference/onnxInference.cpp)
    target_include_directories(AIxeleratorServiceLib PRIVATE ${ONNX_INCLUDE_DIRS})
    target_link_libraries(AIxeleratorServiceLib PUBLIC ${ONNX_LIBRARIES})
endif()

if(WITH_SOL)
    target_sources(AIxeleratorServiceLib PRIVATE inferenceStrategy/solInference/solInference.cpp)
    target_include_directories(AIxeleratorServiceLib PRIVATE ${VEDA_INCLUDE_DIRS})
    target_link_libraries(AIxeleratorServiceLib PUBLIC ${VEDA_LIBRARIES})
endif()

set_target_properties(
    AIxeleratorServiceLib 
    PROPERTIES 
    #    PUBLIC_HEADER 
    #        "aixeleratorService/aixeleratorService.h;aixeleratorService/#aixeleratorService_interfaceC.h;aixeleratorService/#aixeleratorService_interfaceF.f90"
        LINKER_LANGUAGE Fortran
)

install(
    TARGETS AIxeleratorServiceLib
    LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib
    #PUBLIC_HEADER DESTINATION ${CMAKE_BINARY_DIR}/include
)

install(
    DIRECTORY "${CMAKE_SOURCE_DIR}/src/"
    DESTINATION "${CMAKE_BINARY_DIR}/include"
    FILES_MATCHING 
        PATTERN "*.h"
)

install(
    FILES ${CMAKE_SOURCE_DIR}/src/aixeleratorService/aixeleratorService_interfaceF.f90
    DESTINATION "${CMAKE_BINARY_DIR}/include/aixeleratorService"
)
