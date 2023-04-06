#include "inferenceStrategy/solInference/solInference.h"

#include <numeric>
#include <iostream>

#define VEDACHECK(err) veda_check(err, __FILE__, __LINE__)
void veda_check(VEDAresult err, const char* file, const int line) {
    if(err != VEDA_SUCCESS) {
        const char* name = 0;
        vedaGetErrorName(err, &name);
        printf("Error: %i %s @ %s (%i)\n", err, name, file, line);
        assert(false);
        std::exit(1);
    }
}

SOLInference::SOLInference()
{
    input_batch_ = nullptr;
    output_batch_ = nullptr;
}

SOLInference::~SOLInference()
{
    if ( input_batch_ != nullptr )
    {
        VEDACHECK(vedaMemFreeAsync(input_batch_, 0));
    }
    
    if ( output_batch_ != nullptr )
    {
        VEDACHECK(vedaMemFreeAsync(output_batch_, 0));
    }


    VEDACHECK(vedaCtxSynchronize());
    //VEDACHECK(vedaModuleUnload(mod_));
    VEDAresult err;
    try 
    {
        err = vedaCtxDestroy(ctx_);
        veda_check(err, "solInference.cpp", 45);
    }
    catch(const VEDAresult res)
    {
        const char* name = 0;
        vedaGetErrorName(err, &name);
        std::cout << "Error in vedaCtxDestroy: " << res << std::endl;
    }
}

void SOLInference::init(
    int batchsize, 
    int device_id, 
    std::string model_file_name, 
    std::vector<int64_t>& input_shape, double* inputData,  std::vector<int64_t>& output_shape, double* outputData
){
    device_id_ = device_id;
    model_file_name_ = model_file_name;
    batchsize_ = batchsize;
    if (batchsize_ < 1)
    {
        std::cerr << "Error in init SOLInference: batchsize should not be zero or negative!" << std::endl;
    }

    input_ = inputData;
    output_ = outputData;

    int batch_dim = input_shape[0];
    // TODO: generalize for tensors with dim > 2
    // TODO: also remove assumption that input_second_dim == output_second_dim
    int second_dim = input_shape[1];
    num_batches_ = batch_dim / batchsize_;
    size_remaining_ = batch_dim % batchsize_;
    int num_tensors = size_remaining_ > 0 ? num_batches_ + 1 : num_batches_;
    tensor_offsets_.resize(num_tensors);
    for(int i = 0; i < num_tensors; i++)
    {
        // note: second_dim should be used from output shape
        tensor_offsets_[i] = i*batchsize_*second_dim;
    }

    // note: 0 = VEDA_CONTEXT_MODE_OMP
    VEDACHECK(vedaCtxCreate(&ctx_, 0, device_id_));
    VEDACHECK(vedaModuleLoad(&mod_, model_file_name_.c_str()));
    VEDACHECK(vedaModuleGetFunction(&func_, mod_, "predict"));

    std::vector<int64_t> shape = input_shape;
    shape[0] = batchsize_;
    int num_input_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    input_batch_len_ = num_input_elems * sizeof(double);
    VEDACHECK(vedaMemAllocAsync(&input_batch_, input_batch_len_, 0));


    shape = output_shape;
    shape[0] = batchsize_;
    int num_output_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    output_batch_len_ = num_output_elems * sizeof(double);
    VEDACHECK(vedaMemAllocAsync(&output_batch_, output_batch_len_, 0));


    if ( size_remaining_ > 0)
    {
        std::vector<int64_t> input_remainder_shape = input_shape;
        input_remainder_shape[0] = size_remaining_;
        int num_input_remainder_elems = std::accumulate(input_remainder_shape.begin(), input_remainder_shape.end(), 1, std::multiplies<int>());
        input_remainder_len_ = num_input_remainder_elems * sizeof(double);


        std::vector<int64_t> output_remainder_shape = output_shape;
        output_remainder_shape[0] = size_remaining_;
        int num_output_remainder_elems = std::accumulate(output_remainder_shape.begin(), output_remainder_shape.end(), 1, std::multiplies<int>());
        output_remainder_len_ = num_output_remainder_elems * sizeof(double);
    }

    VEDACHECK(vedaCtxSynchronize());
}

void SOLInference::inference()
{
    for ( int i = 0; i < num_batches_; i++ )
    {
        VEDACHECK(vedaMemcpyHtoDAsync(input_batch_, &(input_[tensor_offsets_[i]]), input_batch_len_, 0));  

        VEDACHECK(vedaLaunchKernel(func_, 0, input_batch_, output_batch_));

        VEDACHECK(vedaMemcpyDtoHAsync(&(output_[tensor_offsets_[i]]), output_batch_, output_batch_len_, 0));  
    }

    if ( size_remaining_ > 0)
    {
        VEDACHECK(vedaMemcpyHtoDAsync(input_batch_, &(input_[tensor_offsets_[num_batches_]]), input_remainder_len_, 0));    

        VEDACHECK(vedaLaunchKernel(func_, 0, input_batch_, output_batch_));

        VEDACHECK(vedaMemcpyDtoHAsync(&(output_[tensor_offsets_[num_batches_]]), output_batch_, output_remainder_len_, 0));  
    }

    VEDACHECK(vedaCtxSynchronize());
}

