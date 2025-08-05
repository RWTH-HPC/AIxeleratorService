if(WITH_TORCH)
    set(TORCH_MODEL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/torchModels)
    add_custom_command(
        OUTPUT ${TORCH_MODEL_BINARY_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TORCH_MODEL_BINARY_DIR}
    )
    # create a test model with PyTorch
    set(TORCH_TEST_MODEL ${CMAKE_CURRENT_BINARY_DIR}/torchModels/flexMLP-2x100x100x2.pt)
    set(TORCH_TEST_MODEL_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/torchModels/createModel.py)
    add_custom_command(
        OUTPUT ${TORCH_TEST_MODEL}
        COMMAND ${AIX_VENV_DIR}/bin/python3 ${TORCH_TEST_MODEL_SCRIPT} 2 2 -n 100 100 -o ${TORCH_TEST_MODEL}
        DEPENDS ${TORCH_MODEL_BINARY_DIR} ${TORCH_TEST_MODEL_SCRIPT} 
        COMMENT "Generating PyTorch test model using createModel.py"
        VERBATIM 
    )

    add_custom_target(generate_torch_model ALL
        DEPENDS ${TORCH_TEST_MODEL}
    )

    install(
        FILES ${TORCH_TEST_MODEL}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/models/torchModels
    )
endif()