# AIxeleratorService

A framework to couple HPC applications with ML kernels (e.g. inference) accelerated on GPUs or other devices (e.g. SX-Aurora Tsuabsa).

## Development
Variables to set for intellisense
* `$MPI_INCLUDE` - include directory for MPI (automatically set by MPI modules)
* `$TORCH_INCLUDE` - include directory for libtorch

## Cmake
```
Torch_DIR=<path/to/torch> cmake -DTensorflow_DIR=<path/to/tensorflow> ..
```