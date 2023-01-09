#!/usr/local_rwth/bin/zsh
 
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --job-name=CI-testTorchInference
####SBATCH --output=CI-testTorchInference.%J.out # do not set for CI
#SBATCH --account=rwth0792
#SBATCH --time=00:10:00
#SBATCH --partition=c18g
#SBATCH --gres=gpu:volta:2
#SBATCH --exclusive

module load gcc/10.3
module load cuda/11.4
module load cudnn/8.3.2
module load cmake/3.21.1

rm -rf BUILD-CI
mkdir BUILD-CI && cd BUILD-CI

Torch_DIR=/home/rwth0792/AI-Frameworks/torch/libtorch-1.10.0-cuda-11.3/ cmake .. && \
cmake --build . && \
./test/testTorchInference.cpp.x