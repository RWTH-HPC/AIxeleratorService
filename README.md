# AIxeleratorService

A library to couple HPC applications with ML/DL inference accelerated on GPUs or other devices (e.g. SX-Aurora Tsubasa).

Coupling a traditional HPC application with new approaches using ML/DL inference on heterogeneous architectures can be challenging for application developers.
The goal of this library is to ease the development effort.
For this purpose It hides complexities that arise from the usage of MPI in a distributed heterogeneous environment. 
Moreover it abstracts from the concrete APIs of different ML/DL frameworks (e.g., PyTorch or TensorFlow) by providing a convenient API.
Application developers just need to pass in a (trained) ML/DL model and associated data and the rest will be taken care of by the AIxeleratorService.   

This version of the AIxeleratorService offers different `InferenceMode`s (as defined in `aixeleratorService.h`):
* inference purely on CPUs (`AIX_CPU`)
* inference purely on GPUs (`AIX_GPU`)
* hybrid inference on CPUs + GPUs (`AIX_HYBRID`)

Supported architectures:
* x86 CPUs (tested on Intel Xeon Skylake)
* Nvidia GPUs (tested on V100)
* NEC SX-Aurora Tsubasa Vector Engines (tested on 10B VEs)

Supported machine / deep learning frameworks:
* PyTorch (libtorch)
* TensorFlow
* SOL4VE (only on SX-Aurora)

The AIxelerator Service library is mainly written in C++.   
To support coupling with traditional HPC codes it also offers C and Fortran interface wrappers around its API.

Please note:  
This library is currently in a prototype state. Development will continue and the library will evolve over time with new features being added.
If you are missing any feature to couple your application with this library, please feel free to contact us via: `orland@itc.rwth-aachen.de`

## Installation

### Dependencies
The AIxelerator Service has the following dependencies (tested with version):
* Cmake 3.16.3 or later (3.21.1)
* Compilers
    * C/C++ compiler (Intel 19 + GCC 8 or GCC 10.3.0 or Intel 2021.6.0)
    * Fortran compiler (Intel 19 or Intel 2021.6.0)
* MPI (Intel MPI 2018 or Intel MPI 2021.6)
* CUDA (11.4)
* CUDNN (8.3.2)
* ML/DL frameworks:
    * Torch (1.10.0)
    * TensorFlow (2.6.0 or 2.10.0)
    * SOL4VE (0.4.2.1)

The AIxeleratorService has a modular design that allows to individually decide which ML/DL framework backend for the inference task should be built.
Note that at least **one** ML framework backend is required for a successful build.

### Build with Torch support
To build with Torch backend first download `LibTorch (C++)` from https://pytorch.org. Afterwards your environment should set the variable
```
Torch_DIR=<path/to/torch>
```
to the location of the downloaded distribution of Torch.

### Build with TensorFlow support
To build with TensorFlow backend first download prebuilt TensorFlow C-API from https://www.tensorflow.org/install/lang_c.
Additionally, you will need a (virtual) Python environment with a tensorflow installation.
Afterwards your environment should set the variable
```
Tensorflow_DIR=<path/to/tensorflow>
Tensorflow_Python_DIR=<path/to/tensorflow/python/installation>
```
to the location of the donwloaded distribution of TensorFlow.

### Building the AIxeleratorService
The whole project can easily be built using Cmake.
There are a few important flags to control the build process:
* `WITH_TORCH=<ON|OFF>` enables/disables Torch backend (default: `OFF`)
* `WITH_TENSORFLOW=<ON|OFF>` enables/disables TensorFlow backend (default: `OFF`)
* `Tensorflow_DIR=<path/to/tensorflow>` set TensorFlow install location because there is no `findTensorflow.cmake` yet
* `Tensorflow_Python_DIR=<path/to/tensorflow/python/installation>` TensorFlow Python Installation for generated ProtoBuf C++ classes
* `BUILD_SHARED_LIBS=<ON|OFF>` build AIxelerator Service as shared or static library (default `ON`)
* `BUILD_TESTS=<ON|OFF>` build tests (default: `ON`)

So the complete call to Cmake should look like:
```
mkdir BUILD && cd BUILD
Torch_DIR=<path/to/torch> cmake -DWITH_TORCH=<ON|OFF> -DWITH_TENSORFLOW=<ON|OFF> -DTensorflow_DIR=<path/to/tensorflow> -DBUILD_SHARED_LIBS=<ON|OFF> -DBUILD_TESTS=<ON|OFF> ..
cmake --build .
cmake --install .
```
Afterwards the shared library `libAIxeleratorService.so` can be found in `BUILD/lib`.
Header files are located in `BUILD/include`.
A collection of unit tests can be found in `BUILD/test`.

## Using the AIxeleratorService

### Example in C++
A full minimal working example can be found in `test/testAIxeleratorService.cpp`.
```C++
AIxeleratorService aixelerator(
    model_file, 
    input_shape, input.data(), 
    output_shape, output.data(),
    batch_size
);

aixelerator.inference();
```

### Example in C
A full minimal working example can be found in `test/testAIxeleratorServiceFromC.c`.
```C
AIxeleratorServiceHandle aixelerator = 
    createAIxeleratorService(
        model_file, 
        input_shape, num_input_dims, input_data, 
        output_shape, num_output_dims, output_data, 
        batch_size
    );

inferenceAIxeleratorService(aixelerator);

deleteAIxeleratorService(aixelerator);
```

### Example in Fortran
A full minimal working example can be found in `test/testAIxeleratorServiceFromF.f90`.
```Fortran
aixelerator = createAIxeleratorService_C(model_file_tf, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data, batch_size)

call inferenceAIxeleratorService_C(aixelerator)

call deleteAIxeleratorService_C(aixelerator)
```
