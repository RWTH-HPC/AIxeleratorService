#include "inferenceStrategy/tensorflowInference.h"

#include <iostream>
#include <numeric>
#include <functional>
#include <cstring>

/*
// note(Fabian): only needed if we find a good implementation using TF_NewTensor
void fake_tensor_deallocator(void* data, size_t len, void* arg)
{
    std::string arg_str((char*)arg);
    std::cout << "tensorflowInference: fake tensor deallocator triggered with arg = " << arg_str << "!" << std::endl;
}
*/

TensorflowInference::TensorflowInference()
{

}

TensorflowInference::~TensorflowInference()
{

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
    int num_tensors = size_remaining_ > 0 ? num_batches_ : num_batches_ + 1;
    tensor_offsets_.resize(num_tensors);
    for(int i = 0; i < num_tensors; i++)
    {
        // note: second_dim should be used from output shape
        tensor_offsets_[i] = i*batchsize_*second_dim;
    }

    // TODO: refactor in helper function to avoid redundant code
    // common variables
    std::vector<int64_t> shape;
    int num_elems;

    // allocate InputTensor Batch
    shape = input_shape;
    shape[0] = batchsize_;
    num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    input_batch_len_ = sizeof(double*) * num_elems;
    input_batch_ = TF_AllocateTensor(TF_DOUBLE, shape.data(), shape.size(), input_batch_len_);

    // allocate InputTensor BatchRemainder
    shape = input_shape;
    shape[0] = size_remaining_;
    num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    input_remainder_len_ = sizeof(double*) * num_elems;
    input_remainder_ = TF_AllocateTensor(TF_DOUBLE, shape.data(), shape.size(), input_remainder_len_);

    // allocate OutputTensor Batch
    shape = output_shape;
    shape[0] = batchsize_;
    num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    output_batch_len_ = sizeof(double*) * num_elems;
    output_batch_ = TF_AllocateTensor(TF_DOUBLE, shape.data(), shape.size(), output_batch_len_);

    // allocate OutputTensor BatchRemainder
    shape = output_shape;
    shape[0] = size_remaining_;
    num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    output_remainder_len_ = sizeof(double*) * num_elems;
    output_remainder_ = TF_AllocateTensor(TF_DOUBLE, shape.data(), shape.size(), output_remainder_len_);
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

    app_input_ = input_data;
    app_output_ = output_data;
}

void TensorflowInference::inference()
{
    for( int i = 0; i < num_batches_; i++ )
    {
        double* input_batch_data = (double*) TF_TensorData(input_batch_);
        std::memcpy(input_batch_data, &(app_input_)[tensor_offsets_[i]], input_batch_len_);

        TF_SessionRun(session_, NULL, graph_inputs_, &input_batch_, num_graph_inputs_, graph_outputs_, &output_batch_, num_graph_outputs_, NULL, 0, NULL, status_);
        if (TF_GetCode(status_))
        {
            std::cout << "Tensorflow Run Session Error: " << TF_Message(status_) << std::endl;
        }

        double* output_batch_data = (double*) TF_TensorData(output_batch_);
        // TODO(Fabian): we should find a way to avoid this memcpy
        // but TF always allocates new memory internally in TF_SessionRun()
        std::memcpy(&(app_output_)[tensor_offsets_[i]], output_batch_data, output_batch_len_);
    }

    // remainder inference
    if(size_remaining_ > 0)
    {
        double* input_remainder_data = (double*) TF_TensorData(input_remainder_);  
        std::memcpy(input_remainder_data, &(app_input_)[tensor_offsets_[num_batches_+1]], input_remainder_len_); 

        TF_SessionRun(session_, NULL, graph_inputs_, &input_remainder_, num_graph_inputs_, graph_outputs_, &output_remainder_, num_graph_outputs_, NULL, 0, NULL, status_);
        if (TF_GetCode(status_))
        {
            std::cout << "Tensorflow Run Session (Remainder) Error: " << TF_Message(status_) << std::endl;
        }

        double* output_remainder_data = (double*) TF_TensorData(output_remainder_);
        std::memcpy(&(app_output_)[tensor_offsets_[num_batches_+1]], output_remainder_data, output_remainder_len_);
    }
}
