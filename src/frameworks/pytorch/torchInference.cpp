#include "torchInference.H"
#include <torch/csrc/api/include/torch/utils.h>
//#include <c10/cuda/CUDACachingAllocator.h>
//#include <c10/cuda/CUDAFunctions.h>

#include <iostream>
#include <chrono>

#include <mpi.h>

//using namespace torch::indexing

// deviceNum >= 0  --> GPU device ID
// deviceNum < 0  --> e.g. -1 means no device ==> CPU
torchInference::torchInference( std::string modelFile, int deviceNum )
{
    // Set available num threads to 1
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);

    myDeviceNum_ = deviceNum;

    //!< Load the torch model
    try {
        //!< Deserialize the ScriptModule from a file using torch::jit::load()
        torchModel_ = torch::jit::load(modelFile);

        if ( myDeviceNum_ < 0)
        {
            torchModel_.to(torch::kFloat64); // CPU
        }
        else 
        {
            torchModel_.to(torch::Device(torch::kCUDA, myDeviceNum_)); // GPU
        }     
    }
    catch (const c10::Error& e) {
        std::cerr << "Error in torchInference: Could not load model: " << e.msg() << std::endl;
    }

    // note (Fabian): maybe this should be kCUDA per default for GPU version?
    options_ = torch::TensorOptions(torch::kFloat64);

    // default init to zero-sized tensors
    inputTensor_ = torch::ones({0}, options_);
    outputTensor_ = torch::ones({0}, options_);

    uint batchSize = 100;
    nCellsBatch_ = batchSize;
}

torchInference::~torchInference()
{
}

void torchInference::allocateTensors(std::vector<int> &inputShape, std::vector<int> &outputShape)
{
    //inputTensor_ = torch::ones(inputShape, options_);
    //outputTensor_ = torch::ones(outputShape, options_);
}

void torchInference::batchedForward(int batchsize)
{
    // slice input tensor
    int totalSamples = inputTensor_.sizes()[0];
    int nSlices = totalSamples / batchsize;
    int nCellsRemaining = totalSamples % batchsize;
    if (nCellsRemaining > 0) 
    {
        nSlices++;
    }

    for(int i = 0; i < nSlices; i++)
    {
        // note (Fabian): check for off-by-one-errors
        inputTensorSlice_ = inputTensor_.slice(0, batchsize*i, batchsize*(i+1));
        inputTensorGPU_ = inputTensorSlice_.to(torch::Device(torch::kCUDA, myDeviceNum_));   
        std::vector<torch::jit::IValue> inputTensors = {inputTensorGPU_};   //!< model input needs to be vector of  
        outputTensorGPU_ = torchModel_.forward(inputTensors).toTensor(); 
        outputTensor_.slice(0, batchsize*i, batchsize*(i+1)) = outputTensorGPU_.to(torch::kCPU);
    }
}


void torchInference::forward(double* inputTensor, double* outputTensor, int batchsize)
{
    //! Disabling the autograd functionality to speed up calculations
    //! for more information see: https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/utils.h
    torch::NoGradGuard no_grad;

    batchedForward(batchsize);
}