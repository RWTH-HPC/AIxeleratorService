CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
PROJECT(libFlexMLP_3x2x100x100x2-project)
SET(CMAKE_MODULE_PATH /usr/local/ve/veda/cmake)
SET(CMAKE_BUILD_TYPE Release)
FIND_PACKAGE(VE REQUIRED)
ENABLE_LANGUAGE(VEDA_CXX VEDA_C)
SET(CMAKE_CXX_COMPILER ${CMAKE_VEDA_CXX_COMPILER})
SET(CMAKE_C_COMPILER ${CMAKE_VEDA_C_COMPILER})
INCLUDE_DIRECTORIES(/home/fo014819/anaconda3/envs/sol4ve/lib/python3.7/site-packages/sol/include)
SET(SOURCES
	ve/src/ve_9BC88669_dnn_veblas.cpp
	ve/src/ve_9BC88669_FI.cpp
	ve/src/ve_9BC88669_FI_dfp_ncc_62.cpp
	ve/src/ve_9BC88669_FI_dfp_ncc_69.cpp
	ve/src/ve_9BC88669_FI_dfp_ncc_6D.cpp
	ve/src/ve_9BC88669_ve.cpp
	)
SET(TMPOBJS
	tmp/libFlexMLP_F0_Output.o
	tmp/libFlexMLP_F1_Output.o
	tmp/libFlexMLP_F2_Output.o
	tmp/libFlexMLP_F3_Output.o
	tmp/libFlexMLP_F4_Output.o
	tmp/libFlexMLP_F5_Output.o
	tmp/sol_deployment_runtime.o
	tmp/sol_deployment_runtime_std.o
	)
SET(SHAREDSRC
	shared_lib/src/libFlexMLP.c
	)
SET(MYVEPROFILE
	my_ve_profile.cpp
	)
SET(NCC_LIB /opt/nec/ve/ncc/3.4.0/lib)
SET(NLC_LIB /opt/nec/ve/nlc/2.3.0/lib)
SET(NMPI_LIB /opt/nec/ve/mpi/2.20.0/lib64/ve)
SET(SOL_VE_LIB /home/fo014819/anaconda3/envs/sol4ve/lib/python3.7/site-packages/sol/libve)
LINK_DIRECTORIES(
	 ${NCC_LIB}
	 ${NLC_LIB}
	 ${NMPI_LIB}
	)
ADD_LIBRARY(libFlexMLP_3x2x100x100x2 SHARED ${SOURCES} ${TMPOBJS} ${SHAREDSRC} ${MYVEPROFILE})
TARGET_LINK_LIBRARIES(libFlexMLP_3x2x100x100x2 nc++ ncc ${SOL_VE_LIB}/libsol-backend-veblas-deployment.va cblas blas_sequential)