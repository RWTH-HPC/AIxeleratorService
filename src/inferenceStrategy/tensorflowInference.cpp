#include "inferenceStrategy/tensorflowInference.h"

#include <iostream>
#include <numeric>
#include <functional>
#include <cstring>

void fake_tensor_deallocator(void* data, size_t len, void* arg)
{
    std::string arg_str((char*)arg);
    std::cout << "tensorflowInference: fake tensor deallocator triggered with arg = " << arg_str << "!" << std::endl;
}

void TensorflowInference::initSession()
{
    session_opts_ = TF_NewSessionOptions();
    // we need to get a serialized config to set GPU as there seems to exist no dedicated function in the C API
    // it can be done in a human friendly way using python:
    // https://stackoverflow.com/questions/62393258/tensorflow-c-api-selecting-gpu
    // see also: https://github.com/apivovarov/TF_C_API/blob/master/config.cc
    // for a single GPU, the last value seems to indicate the GPU, counting up from 0x30
    const size_t config_len = 5;
    uint8_t config[config_len] = {0x32, 0x3, 0x2a, 0x1, 0x30 + device_id_};
    TF_SetConfig(session_opts_, (void*)config, config_len, status_);
    const char* tags = "serve";  // must include the set of tags used to identify one MetaGraphDef in the SavedModel
    int num_tags = 1;
    graph_ = TF_NewGraph();
    status_ = TF_NewStatus();
    session_ = TF_LoadSessionFromSavedModel(session_opts_, NULL, model_file_name_.data(), &tags, num_tags, graph_, NULL, status_);

    if ( TF_GetCode(status_) )
    {
        std::cerr << "tensorflowInference: ERROR while loading session." << std::endl;
    }

     // TODO: probably this needs generalization for arbitrary TF-models
    num_graph_inputs_ = num_graph_outputs_ = 1;
    graph_inputs_ = (TF_Output *) malloc(sizeof(TF_Output) * num_graph_inputs_);
    graph_outputs_ = (TF_Output *) malloc(sizeof(TF_Output) * num_graph_outputs_);
    TF_Output i0 = {TF_GraphOperationByName(graph_, "serving_default_input_1"), 0};
    graph_inputs_[0] = i0;
    TF_Output o0 = {TF_GraphOperationByName(graph_, "StatefulPartitionedCall"), 0};
    graph_outputs_[0] = o0;
}

void TensorflowInference::initTensors(
    std::vector<int64_t>& input_shape, double* input_data, 
    std::vector<int64_t>& output_shape, double* output_data
){
    int batch_dim = input_shape[0];
    // TODO: generalize for tensors with dim > 2
    // TODO: also remove assumption that input_second_dim == output_second_dim
    int second_dim = input_shape[1];
    num_batches_ = batch_dim / batchsize_;
    size_remaining_ = batch_dim % batchsize_;
    size_t num_tensors = num_batches_;
    if ( size_remaining_ > 0 )
    {
        num_tensors++;
    }

    // create batch input tensors
    input_tensors_.resize(num_tensors);
    tensor_offsets_.resize(num_tensors);
    tensor_sizes_.resize(num_tensors);
    for( int i = 0; i < num_batches_; i++ )
    {
        std::vector<int64_t> shape = input_shape;
        shape[0] = batchsize_;
        tensor_offsets_[i] = i*batchsize_*second_dim;
        void* data = (void*) &input_data[tensor_offsets_[i]];
        int num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        size_t len = sizeof(double)*num_elems;
        tensor_sizes_[i] = len;
        std::string tensor_name = "inputTensor" + std::to_string(i);
        char* arg = tensor_name.data();
        TF_Tensor* tensor = TF_NewTensor(TF_DOUBLE, shape.data(), shape.size(), data, len, fake_tensor_deallocator, arg);
        if ( tensor == nullptr )
        {
            std::cerr << "Error: could not allocate:" << tensor_name << std::endl;
        }
        else
        {
            input_tensors_[i] = tensor;
        }
    }
    // create input remainder tensor
    if ( size_remaining_ > 0 )
    {
        std::vector<int64_t> shape = input_shape;
        shape[0] = size_remaining_;  
        tensor_offsets_[num_tensors] = num_batches_*batchsize_*second_dim;
        void* data = (void*) &input_data[tensor_offsets_[num_tensors]];  
        int num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        size_t len = sizeof(double)*num_elems;
        tensor_sizes_[num_tensors] = len;
        char* arg = "inputTensorRemainder";
        TF_Tensor* tensor = TF_NewTensor(TF_DOUBLE, shape.data(), shape.size(), data, len, fake_tensor_deallocator, arg);
        if ( tensor == nullptr )
        {
            std::string tensor_name = "inputTensorRemainder";
            std::cerr << "Error: could not allocate:" << tensor_name << std::endl;
        }
        else
        {
            input_tensors_[num_tensors-1] = tensor;
        }
    }

    // create batch output tensors
    output_tensors_.resize(num_tensors);
    for( int i = 0; i < num_batches_; i++ )
    {
        std::vector<int64_t> shape = output_shape;
        shape[0] = batchsize_;
        void* data = (void*) &output_data[i*batchsize_*second_dim];
        int num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        size_t len = sizeof(double)*num_elems;
        std::string tensor_name = "outputTensor" + std::to_string(i);
        char* arg = tensor_name.data();
        TF_Tensor* tensor = TF_NewTensor(TF_DOUBLE, shape.data(), shape.size(), data, len, fake_tensor_deallocator, arg);
        if ( tensor == nullptr )
        {
            std::cerr << "Error: could not allocate:" << tensor_name << std::endl;
        }
        else
        {
            output_tensors_[i] = tensor;
        }
    }
    // create output remainder tensor
    if ( size_remaining_ > 0 )
    {
        std::vector<int64_t> shape = output_shape;
        shape[0] = size_remaining_;  
        void* data = (void*) &output_data[num_batches_*batchsize_*second_dim];  
        int num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        size_t len = sizeof(double)*num_elems;
        char* arg = "outputTensorRemainder";
        TF_Tensor* tensor = TF_NewTensor(TF_DOUBLE, shape.data(), shape.size(), data, len, fake_tensor_deallocator, arg);
        if ( tensor == nullptr )
        {
            std::string tensor_name = "outputTensorRemainder";
            std::cerr << "Error: could not allocate:" << tensor_name << std::endl;
        }
        else
        {
            output_tensors_[num_tensors-1] = tensor;
        }
    }
}

void TensorflowInference::init(
    int batchsize, 
    int device_id, 
    std::string model_file_name, 
    std::vector<int64_t>& input_shape, double* input_data, 
    std::vector<int64_t>& output_shape, double* output_data
){
    batchsize_ = batchsize;
    device_id_ = device_id;
    model_file_name_ = model_file_name;

    initSession();
    initTensors(input_shape, input_data, output_shape, output_data);

    app_output_ = output_data;
}

void TensorflowInference::inference()
{
    int num_tensors = input_tensors_.size();
    for( int i = 0; i < num_tensors; i++ )
    {
        TF_Tensor* input_tensor = input_tensors_[i];
        TF_Tensor* output_tensor = output_tensors_[i];

        TF_SessionRun(session_, NULL, graph_inputs_, &input_tensor, num_graph_inputs_, graph_outputs_, &output_tensor, num_graph_outputs_, NULL, 0, NULL, status_);
        if (TF_GetCode(status_))
        {
            std::cout << "Tensorflow Run Session Error: " << TF_Message(status_) << std::endl;
        }

        double* output_data = (double*) TF_TensorData(output_tensor);

        // TODO(Fabian): we should find a way to avoid this memcpy
        // but TF always allocates new memory internally in SessionRun()
        std::memcpy(&(app_output_)[tensor_offsets_[i]], output_data, tensor_sizes_[i]);
    }
}

/*
void TensorflowInference::inference()
{
    int batch_dim = TF_Dim(input_, 0);
    int second_dim = TF_Dim(input_, 1);    

    double* input_data = (double*) TF_TensorData(input_);
    double* output_data = (double*) TF_TensorData(output_);

    double* input_batch_data  = (double*) TF_TensorData(input_batch_);
    double* output_batch_data = (double*) TF_TensorData(output_batch_);

    // loop over batches
    for( int i = 0; i < num_batches_; i++ )
    {
        // TODO: generalize for more than 2-dimensional tensors
        std::memcpy(input_batch_data, &(input_data)[i*batchsize_*second_dim], sizeof(double) * batch_dim * second_dim);

        TF_SessionRun(session_, NULL, graph_inputs_, &input_batch_, num_graph_inputs_, graph_outputs_, &output_batch_, num_graph_outputs_, NULL, 0, NULL, status_);

        output_batch_data = (double*)TF_TensorData(output_batch_);

        std::memcpy(&(output_data)[i*batchsize_*second_dim], output_batch_data, sizeof(double)*batchsize_*second_dim);
    }

    // remainder loop
    if(size_remaining_ > 0)
    {
        double* input_remainder_data = (double*) TF_TensorData(input_remainder_);
        double* output_remainder_data = (double*) TF_TensorData(output_remainder_);

        std::memcpy(input_remainder_data, &(input_data)[num_batches_*batchsize_*second_dim], sizeof(double) * size_remaining_ * second_dim );

        TF_SessionRun(session_, NULL, graph_inputs_, &input_remainder_, num_graph_inputs_, graph_outputs_, &output_remainder_, num_graph_outputs_, NULL, 0, NULL, status_);

        std::memcpy(&(output_data)[num_batches_*batchsize_*second_dim], output_remainder_data, sizeof(double)*size_remaining_*second_dim);
    }
}
*/