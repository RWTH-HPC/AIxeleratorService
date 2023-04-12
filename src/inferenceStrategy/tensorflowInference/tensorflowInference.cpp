#include "inferenceStrategy/tensorflowInference/tensorflowInference.h"

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

template<typename T>
TensorflowInference<T>::TensorflowInference()
{

}

template<typename T>
TensorflowInference<T>::~TensorflowInference()
{
    // TODO: only GPU-Controller ranks should create TFInference object in AIxeleratorService
    // currently causing a nullptr error
    /*
    TF_DeleteTensor(input_batch_);
    TF_DeleteTensor(output_batch_);

    if( size_remaining_ > 0)
    {
        TF_DeleteTensor(input_remainder_);
        TF_DeleteTensor(output_remainder_);
    }
    */
}

template<typename T>
void TensorflowInference<T>::initSession()
{
    session_opts_ = TF_NewSessionOptions();
    // we need to get a serialized config to set GPU as there seems to exist no dedicated function in the C API
    // it can be done in a human friendly way using python:
    // https://stackoverflow.com/questions/62393258/tensorflow-c-api-selecting-gpu
    // see also: https://github.com/apivovarov/TF_C_API/blob/master/config.cc
    // for a single GPU, the last value seems to indicate the GPU, counting up from 0x30
    // note(fabian): Protobuf generated header: <anaconda_env>/lib/python3.7/site-packages/tensorflow/include/tensorflow/core/protobuf/config.pb.h
    //const size_t config_len = 5;
    //uint8_t config[config_len] = {0x32, 0x3, 0x2a, 0x1, 0x30};
    //config[config_len - 1] = 0x30 + device_id_;

    tensorflow::ConfigProto config;
    config.set_log_device_placement(true);
    auto device_count = config.mutable_device_count();
    if ( device_id_ > -1 )
    {
        std::cout << "TensorflowInference device_id > -1" << std::endl;
        //device_count->insert({"CPU", 0}); // note(fabian): there is a Const op in the model that gets assigned to CPU per default. so if we remove CPU from the device list we get an error.
        device_count->insert({"GPU", 1});

        tensorflow::GPUOptions* gpu_config = config.mutable_gpu_options();
        gpu_config->set_allow_growth(1);
        gpu_config->set_per_process_gpu_memory_fraction(1.0);
        gpu_config->set_visible_device_list(std::to_string(device_id_));
    }
    else
    {
        std::cout << "TensorflowInference device_id < 0" << std::endl;

        device_count->insert({"CPU", 1});
        device_count->insert({"GPU", 0});

        config.set_intra_op_parallelism_threads(1);
        config.set_inter_op_parallelism_threads(1);
    }
    std::string serialized_config;
    if(!config.SerializeToString(&serialized_config))
    {
        std::cout << "ERROR in serializing tensorflow session config" << std::endl;
    }
    TF_SetConfig(session_opts_, serialized_config.c_str(), serialized_config.size(), status_);
    //TF_SetConfig(session_opts_, (void*)config, config_len, status_);
    const char* tags = "serve";  // must include the set of tags used to identify one MetaGraphDef in the SavedModel
    int num_tags = 1;
    graph_ = TF_NewGraph();
    status_ = TF_NewStatus();
    session_ = TF_LoadSessionFromSavedModel(session_opts_, NULL, model_file_name_.data(), &tags, num_tags, graph_, NULL, status_);

    tensorflow::ConfigProto config_proto;

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

template<typename T>
TF_DataType getTypeFromTemplate()
{
    if (std::is_same<T, float>::value)
    {
        return TF_FLOAT;  
    }

    if (std::is_same<T, double>::value)
    {
        return TF_DOUBLE;
    }

    std::cerr << "ERROR: Could not get TF_DataType from template parameter!" << std::endl;
    // TODO(fabian): find a nicer way to handle error. Maybe use std::optional as return type?
    return static_cast<TF_DataType>(-1);
}

template<typename T>
void TensorflowInference<T>::initTensors(
    std::vector<int64_t>& input_shape, T* input_data, 
    std::vector<int64_t>& output_shape, T* output_data
){
    TF_DataType dtype = getTypeFromTemplate<T>();

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

    // TODO: refactor in helper function to avoid redundant code
    // common variables
    std::vector<int64_t> shape;
    int num_elems;

    // allocate InputTensor Batch
    shape = input_shape;
    shape[0] = batchsize_;
    num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    input_batch_len_ = sizeof(T) * num_elems;
    input_batch_ = TF_AllocateTensor(dtype, shape.data(), shape.size(), input_batch_len_);

    // allocate InputTensor BatchRemainder
    shape = input_shape;
    shape[0] = size_remaining_;
    num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    input_remainder_len_ = sizeof(T) * num_elems;
    input_remainder_ = TF_AllocateTensor(dtype, shape.data(), shape.size(), input_remainder_len_);

    // allocate OutputTensor Batch
    shape = output_shape;
    shape[0] = batchsize_;
    num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    output_batch_len_ = sizeof(T) * num_elems;
    output_batch_ = TF_AllocateTensor(dtype, shape.data(), shape.size(), output_batch_len_);

    // allocate OutputTensor BatchRemainder
    shape = output_shape;
    shape[0] = size_remaining_;
    num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    output_remainder_len_ = sizeof(T) * num_elems;
    output_remainder_ = TF_AllocateTensor(dtype, shape.data(), shape.size(), output_remainder_len_);
}


template<typename T>
void TensorflowInference<T>::init(
    int batchsize, 
    int device_id, 
    std::string model_file_name, 
    std::vector<int64_t>& input_shape, T* input_data, 
    std::vector<int64_t>& output_shape, T* output_data
){
    batchsize_ = batchsize;
    if (batchsize_ < 1)
    {
        std::cerr << "Error in init TensorFlowInference: batchsize should not be zero or negative!" << std::endl;
    }
    device_id_ = device_id;
    model_file_name_ = model_file_name;

    initSession();
    initTensors(input_shape, input_data, output_shape, output_data);

    app_input_ = input_data;
    app_output_ = output_data;
}

template<typename T>
void TensorflowInference<T>::inference()
{
    for( int i = 0; i < num_batches_; i++ )
    {
        T* input_batch_data = (T*) TF_TensorData(input_batch_);
        std::memcpy(input_batch_data, &(app_input_)[tensor_offsets_[i]], input_batch_len_);

        TF_SessionRun(session_, NULL, graph_inputs_, &input_batch_, num_graph_inputs_, graph_outputs_, &output_batch_, num_graph_outputs_, NULL, 0, NULL, status_);
        if (TF_GetCode(status_))
        {
            std::cout << "Tensorflow Run Session Error: " << TF_Message(status_) << std::endl;
        }

        T* output_batch_data = (T*) TF_TensorData(output_batch_);
        // TODO(Fabian): we should find a way to avoid this memcpy
        // but TF always allocates new memory internally in TF_SessionRun()
        std::memcpy(&(app_output_)[tensor_offsets_[i]], output_batch_data, output_batch_len_);
    }

    // remainder inference
    if(size_remaining_ > 0)
    {
        T* input_remainder_data = (T*) TF_TensorData(input_remainder_);  
        std::memcpy(input_remainder_data, &(app_input_)[tensor_offsets_[num_batches_]], input_remainder_len_); 

        TF_SessionRun(session_, NULL, graph_inputs_, &input_remainder_, num_graph_inputs_, graph_outputs_, &output_remainder_, num_graph_outputs_, NULL, 0, NULL, status_);
        if (TF_GetCode(status_))
        {
            std::cout << "Tensorflow Run Session (Remainder) Error: " << TF_Message(status_) << std::endl;
        }

        T* output_remainder_data = (T*) TF_TensorData(output_remainder_);
        std::memcpy(&(app_output_)[tensor_offsets_[num_batches_]], output_remainder_data, output_remainder_len_);
    }
}

template class TensorflowInference<float>;
template class TensorflowInference<double>;