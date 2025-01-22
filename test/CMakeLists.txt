if(WITH_TORCH)
    list(APPEND TORCH_TESTS "testTorchInference.cpp")
    foreach(TEST ${TORCH_TESTS})
        message(STATUS "Creating Torch test ${TEST}")
        get_filename_component(UNITTEST_NAME ${TEST} NAME)
        get_filename_component(UNITTEST_DIR ${TEST} DIRECTORY)
        add_executable(${UNITTEST_NAME}.x ${TEST})
        target_sources(
            ${UNITTEST_NAME}.x 
            PRIVATE testTorchInference.cpp 
            ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/torchInference/torchInference.cpp
        )
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${TORCH_INCLUDE_DIRS})
        target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${TORCH_LIBRARIES})
    endforeach(TEST)
endif()


if(WITH_TENSORFLOW)
    list(APPEND TENSORFLOW_TESTS "testTensorflowInference.cpp")
    foreach(TEST ${TENSORFLOW_TESTS})
        message(STATUS "Creating Tensorflow test ${TEST}")
        message(STATUS "Tensorflow include path ${TENSORFLOW_INCLUDE_DIRS}")
        get_filename_component(UNITTEST_NAME ${TEST} NAME)
        get_filename_component(UNITTEST_DIR ${TEST} DIRECTORY)
        add_executable(${UNITTEST_NAME}.x ${TEST})
        target_sources(
            ${UNITTEST_NAME}.x 
            PRIVATE 
            testTensorflowInference.cpp 
            ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInference.cpp
            #${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInferenceCpp.cpp
        )
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
        target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_LIBRARIES})
    endforeach(TEST)
endif()

if(WITH_TENSORFLOW)
    list(APPEND TENSORFLOW_TESTS_CONV2D "testTensorflowInferenceConvolution2D.cpp")
    foreach(TEST ${TENSORFLOW_TESTS_CONV2D})
        message(STATUS "Creating Tensorflow test ${TEST}")
        message(STATUS "Tensorflow include path ${TENSORFLOW_INCLUDE_DIRS}")
        get_filename_component(UNITTEST_NAME ${TEST} NAME)
        get_filename_component(UNITTEST_DIR ${TEST} DIRECTORY)
        add_executable(${UNITTEST_NAME}.x ${TEST})
        target_sources(
            ${UNITTEST_NAME}.x 
            PRIVATE 
            testTensorflowInferenceConvolution2D.cpp 
            ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInference.cpp
            #${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInferenceCpp.cpp
        )
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
        target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_LIBRARIES})
    endforeach(TEST)
endif()

if(WITH_TENSORFLOW)
    list(APPEND TENSORFLOW_TESTS_CONV3D "testTensorflowInferenceConvolution3D.cpp")
    foreach(TEST ${TENSORFLOW_TESTS_CONV3D})
        message(STATUS "Creating Tensorflow test ${TEST}")
        message(STATUS "Tensorflow include path ${TENSORFLOW_INCLUDE_DIRS}")
        get_filename_component(UNITTEST_NAME ${TEST} NAME)
        get_filename_component(UNITTEST_DIR ${TEST} DIRECTORY)
        add_executable(${UNITTEST_NAME}.x ${TEST})
        target_sources(
            ${UNITTEST_NAME}.x 
            PRIVATE 
            testTensorflowInferenceConvolution3D.cpp 
            ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInference.cpp
            #${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInferenceCpp.cpp
        )
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
        target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_LIBRARIES})
    endforeach(TEST)
endif()

if(WITH_TENSORFLOW)
    list(APPEND TENSORFLOW_TESTS_CONV3D_MULTICHANNEL "testTensorflowInferenceConvolution3D_multichannel.cpp")
    foreach(TEST ${TENSORFLOW_TESTS_CONV3D_MULTICHANNEL})
        message(STATUS "Creating Tensorflow test ${TEST}")
        message(STATUS "Tensorflow include path ${TENSORFLOW_INCLUDE_DIRS}")
        get_filename_component(UNITTEST_NAME ${TEST} NAME)
        get_filename_component(UNITTEST_DIR ${TEST} DIRECTORY)
        add_executable(${UNITTEST_NAME}.x ${TEST})
        target_sources(
            ${UNITTEST_NAME}.x 
            PRIVATE 
            testTensorflowInferenceConvolution3D_multichannel.cpp 
            ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInference.cpp
            #${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInferenceCpp.cpp
        )
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
        target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_LIBRARIES})
    endforeach(TEST)
endif()


if(WITH_TENSORFLOW)
    list(APPEND TENSORFLOW_TESTS_TSRGAN "testTensorflowInferenceTSRGAN.cpp")
    foreach(TEST ${TENSORFLOW_TESTS_TSRGAN})
        message(STATUS "Creating Tensorflow test ${TEST}")
        message(STATUS "Tensorflow include path ${TENSORFLOW_INCLUDE_DIRS}")
        get_filename_component(UNITTEST_NAME ${TEST} NAME)
        get_filename_component(UNITTEST_DIR ${TEST} DIRECTORY)
        add_executable(${UNITTEST_NAME}.x ${TEST})
        target_sources(
            ${UNITTEST_NAME}.x 
            PRIVATE 
            testTensorflowInferenceTSRGAN.cpp 
            ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInference.cpp
            #${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInferenceCpp.cpp
        )
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
        target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_LIBRARIES})
    endforeach(TEST)
endif()

if(WITH_TENSORFLOW)
    list(APPEND TENSORFLOW_REPRODUCER "tensorflowReproducer.cpp")
    foreach(TEST ${TENSORFLOW_REPRODUCER})
        message(STATUS "Creating Tensorflow test ${TEST}")
        message(STATUS "Tensorflow include path ${TENSORFLOW_INCLUDE_DIRS}")
        get_filename_component(UNITTEST_NAME ${TEST} NAME)
        get_filename_component(UNITTEST_DIR ${TEST} DIRECTORY)
        add_executable(${UNITTEST_NAME}.x ${TEST})
        target_sources(${UNITTEST_NAME}.x PRIVATE tensorflowReproducer.cpp)
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
        target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_LIBRARIES})
    endforeach(TEST)
endif()

if(WITH_ONNX)
    list(APPEND ONNX_TESTS "testONNXInference.cpp")
    foreach(TEST ${ONNX_TESTS})
        message(STATUS "Creating ONNX test ${TEST}")
        get_filename_component(UNITTEST_NAME ${TEST} NAME)
        get_filename_component(UNITTEST_DIR ${TEST} DIRECTORY)
        add_executable(${UNITTEST_NAME}.x ${TEST})
        target_sources(
                ${UNITTEST_NAME}.x
                PRIVATE testONNXInference.cpp
                ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/onnxInference/onnxInference.cpp
        )
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${ONNX_INCLUDE_DIRS})
        target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${ONNX_LIBRARIES})
    endforeach(TEST)
endif()

list(APPEND DISTRIBUTION_TESTS "testRoundRobinDistribution.cpp")
foreach(TEST ${DISTRIBUTION_TESTS})
    message(STATUS "Creating Distribution test ${TEST}")
    get_filename_component(UNITTEST_NAME ${TEST} NAME)
    get_filename_component(UNITTEST_DIR ${TEST} DIRECTORY)
    add_executable(${UNITTEST_NAME}.x ${TEST})
    target_sources(
        ${UNITTEST_NAME}.x
        PRIVATE 
        testRoundRobinDistribution.cpp 
        ${CMAKE_SOURCE_DIR}/src/distributionStrategy/roundRobinDistribution.cpp
        ${CMAKE_SOURCE_DIR}/src/communicationStrategy/collectiveCommunication.cpp
        ${CMAKE_SOURCE_DIR}/src/communicationStrategy/nonBlockingPtoPCommunication.cpp
    )
    target_include_directories(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_include_directories(${UNITTEST_NAME}.x PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${MPI_LIBRARIES})
endforeach(TEST)


list(APPEND AIXELERATOR_TESTS "testAIxeleratorService.cpp")
foreach(TEST ${AIXELERATOR_TESTS})
    message(STATUS "Creating AIxelerator test ${TEST}")
    get_filename_component(UNITTEST_NAME ${TEST} NAME)
    get_filename_component(UNITTEST_DIR ${TEST} DIRECTORY)
    add_executable(${UNITTEST_NAME}.x ${TEST})
    target_sources(
        ${UNITTEST_NAME}.x 
        PRIVATE 
        ${TEST} 
        ${CMAKE_SOURCE_DIR}/src/aixeleratorService/aixeleratorService.cpp 
        ${CMAKE_SOURCE_DIR}/src/distributionStrategy/roundRobinDistribution.cpp   
        ${CMAKE_SOURCE_DIR}/src/communicationStrategy/collectiveCommunication.cpp 
        ${CMAKE_SOURCE_DIR}/src/communicationStrategy/nonBlockingPtoPCommunication.cpp
    )
    if(WITH_TORCH)
        target_sources(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/torchInference/torchInference.cpp)
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${TORCH_INCLUDE_DIRS})
        target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${TORCH_LIBRARIES})
    endif()
    if(WITH_TENSORFLOW)
        target_sources(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInference.cpp)
        #target_sources(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInferenceCpp.cpp)
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
        target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_LIBRARIES})
    endif()
    if(WITH_ONNX)
        target_sources(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/onnxInference/onnxInference.cpp)
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${ONNX_INCLUDE_DIRS})
        target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${ONNX_LIBRARIES})
    endif()
    if(WITH_SOL)
        target_sources(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/solInference/solInference.cpp)
        target_include_directories(${UNITTEST_NAME}.x PRIVATE ${VEDA_INCLUDE_DIRS})
        target_link_libraries(${UNITTEST_NAME}.x PUBLIC ${VEDA_LIBRARIES})
    endif()
    if(WITH_SCOREP)
        scorep_instrument_target(
            ${UNITTEST_NAME}.x
                USER ON
                COMPILER OFF
                MPP "mpi" 
        )  
    endif()
    target_include_directories(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_include_directories(${UNITTEST_NAME}.x PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${MPI_LIBRARIES})
endforeach(TEST)


message(STATUS "Creating AIxelerator LIB test")
add_executable(testAIxeleratorLib.x testAIxeleratorService.cpp)
target_sources(testAIxeleratorLib.x PRIVATE testAIxeleratorService.cpp)
target_include_directories(testAIxeleratorLib.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(${UNITTEST_NAME}.x PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(testAIxeleratorLib.x PRIVATE ${MPI_LIBRARIES})
if(WITH_TORCH)
    target_include_directories(${UNITTEST_NAME}.x PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(testAIxeleratorLib.x PRIVATE ${TORCH_LIBRARIES})
endif()
if(WITH_TENSORFLOW)
    target_include_directories(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
    target_link_libraries(testAIxeleratorLib.x PRIVATE ${TENSORFLOW_LIBRARIES})
endif()
if(WITH_ONNX)
    target_include_directories(${UNITTEST_NAME}.x PRIVATE ${ONNX_INCLUDE_DIRS})
    target_link_libraries(testAIxeleratorLib.x PRIVATE ${ONNX_LIBRARIES})
endif()
if(WITH_SOL)
    target_include_directories(${UNITTEST_NAME}.x PRIVATE ${VEDA_INCLUDE_DIRS})
    target_link_libraries(${UNITTEST_NAME}.x PUBLIC ${VEDA_LIBRARIES})
endif()
if(WITH_SCOREP)
    scorep_instrument_target(testAIxeleratorLib.x
        USER ON
        COMPILER OFF
        MPP "mpi" 
    )  
endif()
target_link_libraries(testAIxeleratorLib.x PRIVATE AIxeleratorServiceLib)


if(WITH_TORCH)
    message(STATUS "Creating TorchInference C Interface Test")
    add_executable(testTorchInference_interfaceC.x testTorchInferenceFromC.c)
    target_sources(
        testTorchInference_interfaceC.x 
        PRIVATE 
        testTorchInferenceFromC.c 
        ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/torchInference/torchInference.cpp 
        ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/torchInference/torchInference_interfaceC.cpp
    )
    target_include_directories(testTorchInference_interfaceC.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_include_directories(testTorchInference_interfaceC.x PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(testTorchInference_interfaceC.x PRIVATE ${TORCH_LIBRARIES})
endif()


if(WITH_TORCH)
    message(STATUS "Creating TorchInference Fortran Interface Test")
    add_executable(testTorchInference_interfaceF.x testTorchInferenceFromF.f90)
    target_sources(
        testTorchInference_interfaceF.x 
        PRIVATE 
        testTorchInferenceFromF.f90 
        ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/torchInference/torchInference.cpp 
        ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/torchInference/torchInference_interfaceC.cpp 
        ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/torchInference/torchInference_interfaceF.f90
    )
    target_include_directories(testTorchInference_interfaceF.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_include_directories(testTorchInference_interfaceF.x PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(testTorchInference_interfaceF.x PRIVATE ${TORCH_LIBRARIES})
    set_target_properties(testTorchInference_interfaceF.x PROPERTIES LINKER_LANGUAGE Fortran)
endif()


message(STATUS "Creating AIxeleratorService C Interface Test")
add_executable(testAIxeleratorService_interfaceC.x testAIxeleratorServiceFromC.c)
target_sources(
    testAIxeleratorService_interfaceC.x 
    PRIVATE 
    testAIxeleratorServiceFromC.c 
    ${CMAKE_SOURCE_DIR}/src/aixeleratorService/aixeleratorService.cpp 
    ${CMAKE_SOURCE_DIR}/src/aixeleratorService/aixeleratorService_interfaceC.cpp 
    ${CMAKE_SOURCE_DIR}/src/distributionStrategy/roundRobinDistribution.cpp 
    ${CMAKE_SOURCE_DIR}/src/communicationStrategy/collectiveCommunication.cpp
    ${CMAKE_SOURCE_DIR}/src/communicationStrategy/nonBlockingPtoPCommunication.cpp
)
if(WITH_TORCH)
    target_sources(testAIxeleratorService_interfaceC.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/torchInference/torchInference.cpp )
    target_include_directories(testAIxeleratorService_interfaceC.x PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(testAIxeleratorService_interfaceC.x PRIVATE ${TORCH_LIBRARIES})
endif()
if(WITH_TENSORFLOW)
    target_sources(testAIxeleratorService_interfaceC.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInference.cpp)
    #target_sources(testAIxeleratorService_interfaceC.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInferenceCpp.cpp)
    target_include_directories(testAIxeleratorService_interfaceC.x PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
    target_link_libraries(testAIxeleratorService_interfaceC.x PRIVATE ${TENSORFLOW_LIBRARIES})
endif()
if(WITH_ONNX)
    target_sources(testAIxeleratorService_interfaceC.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/onnxInference/onnxInference.cpp )
    target_include_directories(testAIxeleratorService_interfaceC.x PRIVATE ${ONNX_INCLUDE_DIRS})
    target_link_libraries(testAIxeleratorService_interfaceC.x PRIVATE ${ONNX_LIBRARIES})
endif()
if(WITH_SOL)
    target_sources(testAIxeleratorService_interfaceC.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/solInference/solInference.cpp)
    target_include_directories(testAIxeleratorService_interfaceC.x PRIVATE ${VEDA_INCLUDE_DIRS})
    target_link_libraries(testAIxeleratorService_interfaceC.x PUBLIC ${VEDA_LIBRARIES})
endif()
if(WITH_SCOREP)
    scorep_instrument_target(testAIxeleratorService_interfaceC.x
        USER ON
        COMPILER OFF
        MPP "mpi" 
    )  
endif()
target_include_directories(testAIxeleratorService_interfaceC.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(testAIxeleratorService_interfaceC.x PRIVATE  ${MPI_INCLUDE_PATH} )
target_link_libraries(testAIxeleratorService_interfaceC.x PRIVATE  ${MPI_LIBRARIES} )


message(STATUS "Creating AIxeleratorService Fortran Interface Test")
add_executable(testAIxeleratorService_interfaceF.x testAIxeleratorServiceFromF.f90)
target_sources(
    testAIxeleratorService_interfaceF.x 
    PRIVATE 
    testAIxeleratorServiceFromF.f90 
    ${CMAKE_SOURCE_DIR}/src/aixeleratorService/aixeleratorService.cpp 
    ${CMAKE_SOURCE_DIR}/src/aixeleratorService/aixeleratorService_interfaceC.cpp 
    ${CMAKE_SOURCE_DIR}/src/aixeleratorService/aixeleratorService_interfaceF.f90 
    ${CMAKE_SOURCE_DIR}/src/distributionStrategy/roundRobinDistribution.cpp 
    ${CMAKE_SOURCE_DIR}/src/communicationStrategy/collectiveCommunication.cpp
    ${CMAKE_SOURCE_DIR}/src/communicationStrategy/nonBlockingPtoPCommunication.cpp
)
if(WITH_TORCH)
    target_sources(testAIxeleratorService_interfaceF.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/torchInference/torchInference.cpp )
    target_include_directories(testAIxeleratorService_interfaceF.x PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(testAIxeleratorService_interfaceF.x PRIVATE ${TORCH_LIBRARIES})
endif()
if(WITH_TENSORFLOW)
    target_sources(testAIxeleratorService_interfaceF.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInference.cpp)
    #target_sources(testAIxeleratorService_interfaceF.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference/tensorflowInferenceCpp.cpp)
    target_include_directories(testAIxeleratorService_interfaceF.x PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
    target_link_libraries(testAIxeleratorService_interfaceF.x PRIVATE ${TENSORFLOW_LIBRARIES} )
endif()
if(WITH_ONNX)
    target_sources(testAIxeleratorService_interfaceF.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/onnxInference/onnxInference.cpp )
    target_include_directories(testAIxeleratorService_interfaceF.x PRIVATE ${ONNX_INCLUDE_DIRS})
    target_link_libraries(testAIxeleratorService_interfaceF.x PRIVATE ${ONNX_LIBRARIES})
endif()
if(WITH_SOL)
    target_sources(testAIxeleratorService_interfaceF.x PRIVATE ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/solInference/solInference.cpp)
    target_include_directories(testAIxeleratorService_interfaceF.x PRIVATE ${VEDA_INCLUDE_DIRS})
    target_link_libraries(testAIxeleratorService_interfaceF.x PUBLIC ${VEDA_LIBRARIES})
endif()
if(WITH_SCOREP)
    scorep_instrument_target(testAIxeleratorService_interfaceF.x
        USER ON
        COMPILER OFF
        MPP "mpi" 
    )  
endif()
target_include_directories(testAIxeleratorService_interfaceF.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(testAIxeleratorService_interfaceF.x PRIVATE  ${MPI_INCLUDE_PATH} )
target_link_libraries(testAIxeleratorService_interfaceF.x PRIVATE  ${MPI_LIBRARIES} MPI::MPI_Fortran)
set_target_properties(testAIxeleratorService_interfaceF.x PROPERTIES LINKER_LANGUAGE Fortran)

if(WITH_SOL)
    message(STATUS "Creating SOL Inference Test")
    add_executable(testSOLInference.x testSOLInference.cpp)
    target_sources(
        testSOLInference.x 
        PRIVATE
            testSOLInference.cpp
            ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/solInference/solInference.cpp
    )
    target_include_directories(testSOLInference.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_include_directories(testSOLInference.x PRIVATE ${VEDA_INCLUDE_DIRS})
    target_link_libraries(testSOLInference.x PRIVATE ${VEDA_LIBRARIES})
endif()
