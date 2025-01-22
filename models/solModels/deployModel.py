import torch
import sol
from helper import FlexMLP
import argparse
import os
from os import listdir
from stat import *
import shutil

def parseArguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('ninp', type=int,
                        help='Number of input neurons')
    parser.add_argument('nout', type=int,
                        help='Number of output neurons')
    parser.add_argument('-n', '--neurons', dest='neurons', nargs='+', default=[32, 32],
                        help='List of neurons in hidden layers, e.g. -n 32 64 32')
    parser.add_argument('--output', '-o', dest='model', required=False, default='Script.pt',
                        help='File name of save torch script, default: Script.pt')
    parser.add_argument('--with-float', dest='withFloat', default=False, action='store_true')
    parser.add_argument('-bs', '--batchsize', type=int, dest='batchsize', required=False, default=1)
    parser.add_argument('--sol-include-path', dest='solIncludePath', required=True, help='Example: /home/HPCACCOUNT/anaconda3/envs/sol4ve/lib/python3.7/site-packages/sol/include')
    parser.add_argument('--sol-ve-lib', dest='solVELib', required=True, help='Example: /home/HPCACCOUNT/anaconda3/envs/sol4ve/lib/python3.7/site-packages/sol/libve')

    return parser.parse_args()

def createMyVeProfileDummy(path):
    my_ve_profile_content = [
        "// needs to be compiled with nc++ (.vcpp ending does not work for some reason)",
        "#include <functional>",
        "#include <cstdint>",
        "uint64_t sol_ve_profile(std::function<void(void)> func, const int _runs, const int _notImproved){return 0;}",
        "void sol_check(int, char const*, int) {}"
    ]
    with open(path + "/my_ve_profile.cpp", 'w') as my_ve_profile_file: 
        for line in my_ve_profile_content:
            my_ve_profile_file.write(line + "\n")

def createCMake(libName, solIncludePath, solVeLibPath):
    # configurable parameters
    projectName = libName + "-project"
    cmakeBuildType = "Release"
    solInclude = solIncludePath
    # TODO: find a nicer way to automatically determine compiler version and path
    nccLib = "/opt/nec/ve/ncc/3.4.0/lib"
    nlcLib = "/opt/nec/ve/nlc/2.3.0/lib"
    nmpiLib = "/opt/nec/ve/mpi/2.20.0/lib64/ve"
    solVeLib = solVeLibPath

    with open('./' + libName + '/CMakeLists.txt', 'w') as cmakeFile:
        cmakeFile.write("CMAKE_MINIMUM_REQUIRED(VERSION 3.13)\n")
        cmakeFile.write(f"PROJECT({projectName})\n")
        cmakeFile.write("SET(CMAKE_MODULE_PATH /usr/local/ve/veda/cmake)\n")
        cmakeFile.write(f"SET(CMAKE_BUILD_TYPE {cmakeBuildType})\n")
        cmakeFile.write("FIND_PACKAGE(VE REQUIRED)\n")
        cmakeFile.write("ENABLE_LANGUAGE(VEDA_CXX VEDA_C)\n")
        cmakeFile.write("SET(CMAKE_CXX_COMPILER ${CMAKE_VEDA_CXX_COMPILER})\n")
        cmakeFile.write("SET(CMAKE_C_COMPILER ${CMAKE_VEDA_C_COMPILER})\n")
        cmakeFile.write(f"INCLUDE_DIRECTORIES({solInclude})\n")
        
        # find names of genereated VE sources
        # TODO: remove include only files
        srcFiles = [f for f in listdir("./" + libName + "/ve/src") if f.endswith(".cpp")]
        srcFiles.sort(key=str.lower)
        cmakeFile.write("SET(SOURCES\n")
        for file in srcFiles:
            # remove files that only contain an include of another file
            if not (file.endswith("_0.cpp") or file.endswith("_1.cpp") or file.endswith("_2.cpp")):
                cmakeFile.write(f"\tve/src/{file}\n")
        cmakeFile.write("\t)\n")

        # find names of generated object files
        tmpObjFiles = [f for f in listdir("./" + libName + "/tmp") if f.endswith(".o")]
        tmpObjFiles.sort(key=str.lower)
        cmakeFile.write("SET(TMPOBJS\n")
        for file in tmpObjFiles:
            cmakeFile.write(f"\ttmp/{file}\n")
        cmakeFile.write("\t)\n")

        # find name of forward wrapper C file
        sharedSrcFiles = [f for f in listdir("./" + libName + "/shared_lib/src") if f.endswith(".c") and not f.endswith("example.c")]
        cmakeFile.write("SET(SHAREDSRC\n")
        for file in sharedSrcFiles:
            cmakeFile.write(f"\tshared_lib/src/{file}\n")
        cmakeFile.write("\t)\n")

        cmakeFile.write("SET(MYVEPROFILE\n")
        cmakeFile.write("\tmy_ve_profile.cpp\n")
        cmakeFile.write("\t)\n")

        cmakeFile.write(f"SET(NCC_LIB {nccLib})\n")
        cmakeFile.write(f"SET(NLC_LIB {nlcLib})\n")
        cmakeFile.write(f"SET(NMPI_LIB {nmpiLib})\n")
        cmakeFile.write(f"SET(SOL_VE_LIB {solVeLib})\n")
        cmakeFile.write("LINK_DIRECTORIES(\n")
        cmakeFile.write("\t ${NCC_LIB}\n")
        cmakeFile.write("\t ${NLC_LIB}\n")
        cmakeFile.write("\t ${NMPI_LIB}\n")
        cmakeFile.write("\t)\n")
        cmakeFile.write(f"ADD_LIBRARY({libName} " + "SHARED ${SOURCES} ${TMPOBJS} ${SHAREDSRC} ${MYVEPROFILE})\n")
        cmakeFile.write(f"TARGET_LINK_LIBRARIES({libName} nc++ ncc " + "${SOL_VE_LIB}/libsol-backend-veblas-deployment.va" + " cblas blas_sequential)")

def generateLibName(args):
    libName = "libFlexMLP_" + str(args.batchsize) + "x" + str(args.ninp)
    for neurons in args.neurons:
        libName += "x" + str(neurons)
    libName += "x" + str(args.nout)
    return libName

# replace occurences of "sol_external_ptr" with "sol_internal_ptr"
def fixSolExternalPtr(libName):
    for file in listdir("./" + libName + "/ve/src"):
        if file.endswith(".cpp"):
            with open("./" + libName + "/ve/src/" + file, 'r') as curFile:
                fileContent = curFile.readlines()
            with open("./" + libName + "/ve/src/" + file, 'w') as curFile:
                for line in fileContent:
                    substring = "sol_external_ptr"
                    if substring in line:
                        start = line.find(substring)
                        substringLen = len(substring)
                        firstPart = line[0:start]
                        lastPart = line[start+substringLen:]
                        line = firstPart + "sol_internal_ptr" + lastPart
                    curFile.write(line)

# one external malloc for last layer that should not do anything because memory is already allocated by host code
def fixSolExternalMalloc(libName):
    for file in listdir("./" + libName + "/ve/src"):
        if file.endswith("_FI.cpp"):
            with open("./" + libName + "/ve/src/" + file, 'r') as bugFile:
                fileContent = bugFile.readlines()
            with open("./" + libName + "/ve/src/" + file, 'w') as bugFile:
                for line in fileContent:
                    if "sol_external_malloc" in line:
                        line = "//" + line
                    bugFile.write(line)

def fixSolDeployment(libName, solIncludePath, solVeLibPath):
    if os.path.isdir("./" + libName):
    #mode = os.stat("./" + libName).st_mode
    #if S_ISDIR(mode):
        shutil.rmtree("./" + libName)
    os.rename(".sol", libName)
    createMyVeProfileDummy("./" + libName)
    createCMake(libName, solIncludePath, solVeLibPath)
    fixSolExternalPtr(libName)
    fixSolExternalMalloc(libName)

def createMyWrapperCode(wrapper_dir, header_name):
    my_wrapper_file_content = [
        "#include <veda_device.h>",
        "#include <stdlib.h>",
        "#include <stdio.h>",
        f"#include \"./{header_name}.h\" // the generated header file",
        "",
        "#define CHECK(err) check(err, __FILE__, __LINE__)",
        "",
        "void check(VEDAresult err, const char* file, const int line) {",
        "  if(err != VEDA_SUCCESS) {",
        "    const char* name = 0;",
        "    vedaGetErrorName(err, &name);",
        "    printf(\"Error: %i %s @ %s (%i)\\n\", err, name, file, line);", 
        "    assert(false);",
        "    exit(1);",
        "  }",
        "}",
        "",
        "// VEDA 0.10.2",
        "extern \"C\" void predict(VEDAdeviceptr input_, VEDAdeviceptr output_) {",
        "const double* input = VEDAptr<double>(input_).ptr();",
        "const double* rawInput;",
        "vedaMemPtr(&rawInput, input_);",
        "",
        "double* output = VEDAptr<double>(output_).ptr();",
        "double* rawOutput;",
        "vedaMemPtr(&rawOutput, output_);",
        "",
        "forward(0, rawInput, rawOutput);",
        "}",
    ]

    with open(wrapper_dir + '/my_wrapper.vcpp', 'w') as my_wrapper_file:
        for line in my_wrapper_file_content:
            my_wrapper_file.write(line + "\n")



def createWrapperCMake(wrapper_dir, libName):
    target_name = libName + "_veda_wrapper"
    my_wrapper_cmake_content = [
        "CMAKE_MINIMUM_REQUIRED(VERSION 3.13)",
        f"PROJECT({target_name} LANGUAGES CXX)",
        "SET(CMAKE_MODULE_PATH /usr/local/ve/veda/cmake)",
        "FIND_PACKAGE(VE)",
        "ENABLE_LANGUAGE(VEDA_CXX)",
        "SET(CMAKE_BUILD_TYPE Release)",
        "INCLUDE_DIRECTORIES(${VEDA_INCLUDES})",
        f"ADD_LIBRARY({target_name}" + " SHARED ${CMAKE_CURRENT_LIST_DIR}/my_wrapper.vcpp)",
        f"TARGET_COMPILE_OPTIONS({target_name}" + " PUBLIC \"-O4\" \"-finline-functions\")",
        f"SET(FLEXMLP_LIB {libName})",
        f"TARGET_LINK_LIBRARIES({target_name}" + " ${CMAKE_CURRENT_LIST_DIR}/../BUILD/lib${FLEXMLP_LIB}.so ${VEDA_DEVICE_LIBRARY})",
        "ADD_LIBRARY(veda-mwe-device SHARED my_wrapper.vcpp)"
    ]

    with open(wrapper_dir + '/CMakeLists.txt', 'w') as cmakeFile:
        for line in my_wrapper_cmake_content:
            cmakeFile.write(line + "\n")

def main():
    args = parseArguments()

    n_inp = args.ninp
    n_out = args.nout
    n_neurons = [int(x) for x in args.neurons]
    n_batch = args.batchsize
    print(f"Got batchsize = {n_batch}")

    # requires_grad is needed for SOL to export buffers correctly
    if args.withFloat:
        data_type = torch.float
        m = FlexMLP(n_inp, n_out, n_neurons).requires_grad_(True)
    else:
        data_type = torch.double
        m = FlexMLP(n_inp, n_out, n_neurons).double().requires_grad_(True)

    s = torch.jit.script(m)
    s.save(args.model)
        
    # create dummy input
    input = torch.ones(n_batch, n_inp, dtype=data_type)

    # arguments for SOL deployment
    dargs = {
        "lib_name": "libFlexMLP", 
        "func_name": "forward", 
        "path": "./libFlexMLP", 
        "device_type": "ve"
    }
    sol.deploy(m, "shared_lib", dargs, input)
    libName = generateLibName(args)
    fixSolDeployment(libName, args.solIncludePath, args.solVELib)

    # move scripted PyTorch model into deployed lib folder for reference
    model_file = "./" + args.model
    if os.path.exists(model_file):
        shutil.move(model_file, "./" + libName)

    # create wrapper directory inside deployed lib folder
    wrapper_dir = "./" + libName + "/" + "wrapper"
    os.mkdir(wrapper_dir)
    lib_header = dargs["path"] + "/" + dargs["lib_name"] + ".h"
    if os.path.exists(lib_header):
        shutil.move(lib_header, wrapper_dir)
    createMyWrapperCode(wrapper_dir, dargs["lib_name"])
    createWrapperCMake(wrapper_dir, libName)

    # remove the usual lib folder generated by SOL
    default_sol_deploy_dir = dargs["path"]
    if os.path.exists(default_sol_deploy_dir):
        shutil.rmtree(default_sol_deploy_dir)

    

if __name__ == "__main__":
    main()