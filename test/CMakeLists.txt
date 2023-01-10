list(APPEND TORCH_TESTS "testTorchInference.cpp")

foreach(TEST ${TORCH_TESTS})
    message(STATUS "Creating Torch test ${TEST}")
    get_filename_component(UNITTEST_NAME ${TEST} NAME)
    get_filename_component(UNITTEST_DIR ${TEST} DIRECTORY)
    add_executable(${UNITTEST_NAME}.x ${TEST})
    target_sources(${UNITTEST_NAME}.x PRIVATE testTorchInference.cpp ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/torchInference.cpp)
    target_include_directories(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_include_directories(${UNITTEST_NAME}.x PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${TORCH_LIBRARIES})
endforeach(TEST)


list(APPEND TENSORFLOW_TESTS "testTensorflowInference.cpp")

foreach(TEST ${TENSORFLOW_TESTS})
    message(STATUS "Creating Tensorflow test ${TEST}")
    message(STATUS "Tensorflow include path ${TENSORFLOW_INCLUDE_DIRS}")
    get_filename_component(UNITTEST_NAME ${TEST} NAME)
    get_filename_component(UNITTEST_DIR ${TEST} DIRECTORY)
    add_executable(${UNITTEST_NAME}.x ${TEST})
    target_sources(${UNITTEST_NAME}.x PRIVATE testTensorflowInference.cpp ${CMAKE_SOURCE_DIR}/src/inferenceStrategy/tensorflowInference.cpp)
    target_include_directories(${UNITTEST_NAME}.x PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_include_directories(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
    target_link_libraries(${UNITTEST_NAME}.x PRIVATE ${TENSORFLOW_LIBRARIES})
endforeach(TEST)

