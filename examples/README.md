module load intel intelmpi cuda/11.4 cudnn/8.3.2 gcc/8

mkdir BUILD && cd BUILD
TorchDIR=Torch_DIR=/home/rwth0792/AI-Frameworks/torch/libtorch-1.10.0-cuda-11.3 cmake ..
cmake --build .