#!/bin/zsh

set -x

cd /home/rwth0792/CIAO-AI-MAIN/aixeleratorservice
rm -rf BUILD
rm -rf INSTALL

mkdir BUILD && mkdir INSTALL && cd BUILD
# Torch only
# VERBOSE=1 cmake .. -DWITH_TORCH=ON -DTORCH_VERSION="2.7.1" -DCMAKE_INSTALL_PREFIX=/home/rwth0792/CIAO-AI-MAIN/aixeleratorservice/INSTALL && \
# TensorFlow only
# VERBOSE=1 cmake .. -DWITH_TENSORFLOW=ON -DTENSORFLOW_VERSION=2.17.0 -DCMAKE_INSTALL_PREFIX=/home/rwth0792/CIAO-AI-MAIN/aixeleratorservice/INSTALL && \
# ONNX only
VERBOSE=1 cmake .. -DWITH_ONNX=ON -DONNX_VERSION=1.22.0 -DCMAKE_INSTALL_PREFIX=/home/rwth0792/CIAO-AI-MAIN/aixeleratorservice/INSTALL && \
VERBOSE=1 cmake --build . -j && cmake --install .

# ldd src/libAIxeleratorService.so
# ldd test/testTensorflowInferenceConvolution2D.cpp.x

# ldd ../INSTALL/lib/libAIxeleratorService.so
# ldd ../INSTALL/bin/testTensorflowInferenceConvolution2D.cpp.x

cd ../INSTALL
