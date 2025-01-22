#!/usr/local_rwth/bin/zsh
 
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
####SBATCH --mem-per-cpu=3900M
#SBATCH --mem=100G
#SBATCH --job-name=CI-testAIxeleratorService
####SBATCH --output=CI-testAIxeleratorService.%J.out # do not set for CI
#SBATCH --account=rwth0792
#SBATCH --time=00:10:00
#SBATCH --partition=c18g
#SBATCH --gres=gpu:volta:2
#SBATCH --exclusive

module load gcc/10.3
module load cuda/11.4
module load cudnn/8.3.2
module load cmake/3.21.1

echo ${HOST}
echo ${HOSTNAME}

echo ${LD_LIBRARY_PATH}

pwd

cd /home/rwth0792/aixeleratorservice

rm -rf BUILD-CI-AIX
mkdir BUILD-CI-AIX && cd BUILD-CI-AIX

Torch_DIR=/home/rwth0792/AI-Frameworks/torch/libtorch-1.10.0-cuda-11.3/ cmake -DTensorflow_DIR=/home/rwth0792/AI-Frameworks/libtensorflow-gpu-linux-x86_64-2.6.0 .. && \
cmake --build . && \
CUDA_VISIBLE_DEVICES=1 mpirun ${FLAGS_MPI_BATCH} ./test/testAIxeleratorService.cpp.x ../models/torchModels/flexMLP-2x100x100x2.pt