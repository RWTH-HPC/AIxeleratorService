#!/bin/zsh

#module purge
#module load GCC/8.3.0
#module load intel-compilers/2022.2.1
#module load impi/2021.7.1
#module load cuDNN/8.2.1.32-CUDA-11.3.1
#module load CUDA/11.3.1
#module load GCCcore/.11.3.0
#module load CMake/3.21.1

module purge
module load intel/2022a
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0
module load CMake/3.26.3
module load Score-P/8.1