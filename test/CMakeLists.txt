add_executable(testAIModelTypeErasure
    models/testAIModelTypeErasure.cpp              
)       

target_include_directories(testAIModelTypeErasure PRIVATE
    ${CMAKE_SOURCE_DIR}/src/models
)  

add_executable(testAIModelInheritence
    models/testAIModelInheritence.cpp              
)   

target_include_directories(testAIModelInheritence PRIVATE
    ${CMAKE_SOURCE_DIR}/src/models
)  

add_executable(testAIModel
    models/testAIModel.cpp              
)   

target_include_directories(testAIModel PRIVATE
    ${CMAKE_SOURCE_DIR}/src/models
)  

target_link_libraries(testAIModel PRIVATE
    AIxelerator
)  

