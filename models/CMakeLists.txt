if(WITH_TORCH)
    # create folder structure: ${CMAKE_BINARY_DIR}/models/torchModels
    set(TORCH_MODEL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/torchModels)
    add_custom_command(
        OUTPUT ${TORCH_MODEL_BINARY_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TORCH_MODEL_BINARY_DIR}
    )
    # create a test model with PyTorch
    set(TORCH_TEST_MODEL ${TORCH_MODEL_BINARY_DIR}/flexMLP-2x100x100x2.pt)
    set(TORCH_TEST_MODEL_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/torchModels/createModel.py)
    add_custom_command(
        OUTPUT ${TORCH_TEST_MODEL}
        COMMAND ${AIX_VENV_DIR}/bin/python3 ${TORCH_TEST_MODEL_SCRIPT} 2 2 -n 100 100 -o ${TORCH_TEST_MODEL}
        DEPENDS ${TORCH_MODEL_BINARY_DIR} ${TORCH_TEST_MODEL_SCRIPT} 
        COMMENT "Generating PyTorch test model using createModel.py"
        VERBATIM 
    )

    add_custom_target(GENERATE_TORCH_MODEL ALL
        DEPENDS ${TORCH_TEST_MODEL}
    )

    install(
        FILES ${TORCH_TEST_MODEL}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/models/torchModels
    )
endif()

if(WITH_TENSORFLOW)
    # create folder structure: ${CMAKE_BINARY_DIR}/models/tensorflowModels
    set(TENSORFLOW_MODEL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflowModels)
    add_custom_command(
        OUTPUT ${TENSORFLOW_MODEL_BINARY_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TENSORFLOW_MODEL_BINARY_DIR}
    )

    # create FlexMLP test model
    set(TENSORFLOW_FLEXMLP_MODEL ${TENSORFLOW_MODEL_BINARY_DIR}/flexMLP-2x100x100x2.tf)
    set(TENSORFLOW_FLEXMLP_MODEL_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/tensorflowModels/createModel_tf.py)
    add_custom_command(
        OUTPUT ${TENSORFLOW_FLEXMLP_MODEL}
        COMMAND ${AIX_VENV_DIR}/bin/python3 ${TENSORFLOW_FLEXMLP_MODEL_SCRIPT} 2 2 -n 100 100 -o ${TENSORFLOW_FLEXMLP_MODEL}
        WORKING_DIRECTORY ${TENSORFLOW_MODEL_BINARY_DIR}
        DEPENDS ${TENSORFLOW_MODEL_BINARY_DIR} ${TENSORFLOW_FLEXMLP_MODEL_SCRIPT} 
        COMMENT "Generating TensorFlow FlexMLP-2x100x100x2.tf test model using createModel_tf.py"
        VERBATIM 
    )
    add_custom_target(GENERATE_TENSORFLOW_FLEXMLP_MODEL ALL
        DEPENDS ${TENSORFLOW_FLEXMLP_MODEL}
    )

    # create FlexCNN2D test model
    set(TENSORFLOW_CONV2D_MODEL ${TENSORFLOW_MODEL_BINARY_DIR}/testConvolution2D.tf)
    set(TENSORFLOW_CONV2D_MODEL_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/tensorflowModels/createFlexCNN2D.py)
    add_custom_command(
        OUTPUT ${TENSORFLOW_CONV2D_MODEL}
        COMMAND ${AIX_VENV_DIR}/bin/python3 ${TENSORFLOW_CONV2D_MODEL_SCRIPT}
        WORKING_DIRECTORY ${TENSORFLOW_MODEL_BINARY_DIR}
        DEPENDS ${TENSORFLOW_MODEL_BINARY_DIR} ${TENSORFLOW_CONV2D_MODEL_SCRIPT} 
        COMMENT "Generating TensorFlow testConvolution2D.tf test model using createFlexCNN2D.py"
        VERBATIM 
    )
    add_custom_target(GENERATE_TENSORFLOW_CONV2D_MODEL ALL
        DEPENDS ${TENSORFLOW_CONV2D_MODEL}
    )

    # create FlexCNN3D test model
    set(TENSORFLOW_CONV3D_MODEL ${TENSORFLOW_MODEL_BINARY_DIR}/testConvolution3D.tf)
    set(TENSORFLOW_CONV3D_MODEL_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/tensorflowModels/createFlexCNN3D.py)
    add_custom_command(
        OUTPUT ${TENSORFLOW_CONV3D_MODEL}
        COMMAND ${AIX_VENV_DIR}/bin/python3 ${TENSORFLOW_CONV3D_MODEL_SCRIPT}
        WORKING_DIRECTORY ${TENSORFLOW_MODEL_BINARY_DIR}
        DEPENDS ${TENSORFLOW_MODEL_BINARY_DIR} ${TENSORFLOW_CONV3D_MODEL_SCRIPT} 
        COMMENT "Generating TensorFlow testConvolution3D.tf test model using createFlexCNN3D.py"
        VERBATIM 
    )
    add_custom_target(GENERATE_TENSORFLOW_CONV3D_MODEL ALL
        DEPENDS ${TENSORFLOW_CONV3D_MODEL}
    )

    # note: tensorflow saved models are directories!
    install(
        DIRECTORY ${TENSORFLOW_FLEXMLP_MODEL} ${TENSORFLOW_CONV2D_MODEL} ${TENSORFLOW_CONV3D_MODEL}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/models/tensorflowModels
    )
endif()