#include "inferenceStrategy/torchInference/torchInference.h"

#include <vector>
#include <memory>
#include <iostream>

#include <ATen/ATen.h>

void TorchInference::init(
    int batchsize, 
    int device_id, 
    std::string model_file_name, 
    std::vector<int64_t>& input_shape, double* inputData,  std::vector<int64_t>& output_shape, double* outputData
){
    device_id_ = device_id;
    model_file_name_ = model_file_name;
    try{
        torch_model_ = torch::jit::load(model_file_name_);

        // device IDs from 0 to (n-1) represent n GPUs
        if (device_id_ > -1)
        {
            torch_model_.to(torch::Device(torch::kCUDA, device_id_));
        }
        else 
        {   
            // TODO: datatype needs to be templated
            torch_model_.to(torch::kFloat64);
        }
    }
    catch (const c10::Error& e) {
        throw;
    }

    batchsize_ = batchsize;
    if (batchsize_ < 1)
    {
        std::cerr << "Error in init TorchInference: batchsize should not be zero or negative!" << std::endl;
    }

    const torch::TensorOptions options(torch::kFloat64);
    
    at::IntArrayRef input_sizes = input_shape;
    input_ = torch::from_blob((void*) inputData, input_sizes, options);

    at::IntArrayRef output_sizes = output_shape;
    output_ = torch::from_blob((void*) outputData, output_sizes, options);
}

void TorchInference::inference()
{
    int batch_dim = input_.size(0);
    int num_batches = batch_dim / batchsize_;
    int size_remaining = batch_dim % batchsize_;
    if (size_remaining > 0)
    {
        num_batches++;
    }

    if (device_id_ > -1)
    {
        for( int i = 0; i < num_batches; i++)
        {
            input_batch_ = input_.slice(0, batchsize_*i, batchsize_*(i+1));
            input_gpu_ = input_batch_.to(torch::Device(torch::kCUDA, device_id_));
            std::vector<torch::jit::IValue> inputs = {input_gpu_};
            output_gpu_ = torch_model_.forward(inputs).toTensor();
            output_.slice(0, batchsize_*i, batchsize_*(i+1)) = output_gpu_.to(torch::kCPU);
        }
    }
    else
    {
        for( int i = 0; i < num_batches; i++)
        {
            input_batch_ = input_.slice(0, batchsize_*i, batchsize_*(i+1));
            std::vector<torch::jit::IValue> inputs = {input_batch_};
            output_batch_ = torch_model_.forward(inputs).toTensor();
            output_.slice(0, batchsize_*i, batchsize_*(i+1)) = output_batch_.to(torch::kCPU);
        }   
    }
}
