include(CTest)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

list(APPEND AIXELERATOR_UNIT_TESTS "models/test_AIModel.cpp"
     "core/test_data_distributor.cpp" "core/test_adapter.cpp")

include(GoogleTest)

foreach(TEST ${AIXELERATOR_UNIT_TESTS})
  message(STATUS "Creating test ${TEST}")
  get_filename_component(UNITTEST_NAME ${TEST} NAME)
  get_filename_component(UNITTEST_DIR ${TEST} DIRECTORY)
  add_executable(${UNITTEST_NAME}.x ${TEST})
  target_link_libraries(${UNITTEST_NAME}.x PRIVATE GTest::gtest_main
                                                   AIxelerator)
  gtest_discover_tests(
    ${UNITTEST_NAME}.x
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/${UNITTEST_DIR})
endforeach(TEST)

# add_executable(test_AIModel.x models/testAIModel.cpp )

# target_link_libraries( test_AIModel.x GTest::gtest_main AIxelerator )
