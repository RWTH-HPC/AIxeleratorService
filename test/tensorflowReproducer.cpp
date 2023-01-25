#include <iostream>

#include <tensorflow/c/c_api.h>

#include <vector>

#include <numeric>
#include <functional>

void fake_tensor_deallocator(void* data, size_t len, void* arg)
{
    std::string arg_str((char*)arg);
    std::cout << "fake tensor deallocator triggered with arg = " << arg_str << "!" << std::endl;
}

void initSession(TF_Graph*& graph, TF_Session*& session, TF_Output*& graph_inputs, TF_Output*& graph_outputs, int num_graph_inputs, int num_graph_outputs, int device_id)
{
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    // we need to get a serialized config to set GPU as there seems to exist no dedicated function in the C API
    // it can be done in a human friendly way using python:
    // https://stackoverflow.com/questions/62393258/tensorflow-c-api-selecting-gpu
    // see also: https://github.com/apivovarov/TF_C_API/blob/master/config.cc
    // for a single GPU, the last value seems to indicate the GPU, counting up from 0x30
    //const size_t config_len = 5;
    //uint8_t config[config_len] = {0x32, 0x3, 0x2a, 0x1, 0x30 + device_id};
    //TF_SetConfig(session_opts, (void*)config, config_len, status);
    const size_t config_len = 18;
    uint8_t config[config_len] = {0xa, 0x7, 0xa, 0x3, 0x43, 0x50, 0x55, 0x10, 0x1, 0xa, 0x7, 0xa, 0x3, 0x47, 0x50, 0x55, 0x10, 0x0};
    TF_SetConfig(session_opts, (void*)config, config_len, status);
    //const char* model_file = model_file_name_.data();
    const char* model_file = "../models/tensorflowModels/flexMLP-2x100x100x2.tf";
    const char* tags = "serve";  // must include the set of tags used to identify one MetaGraphDef in the SavedModel
    int num_tags = 1;
    graph = TF_NewGraph();
   
    session = TF_LoadSessionFromSavedModel(session_opts, NULL, model_file, &tags, num_tags, graph, NULL, status);

    if ( TF_GetCode(status) )
    {
        std::cerr << "tensorflowInference: ERROR while loading session." << std::endl;
    }

     // TODO: probably this needs generalization for arbitrary TF-models
    graph_inputs = (TF_Output *) malloc(sizeof(TF_Output) * num_graph_inputs);
    graph_outputs = (TF_Output *) malloc(sizeof(TF_Output) * num_graph_outputs);
    TF_Output i0 = {TF_GraphOperationByName(graph, "serving_default_input_1"), 0};
    graph_inputs[0] = i0;
    TF_Output o0 = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};
    graph_outputs[0] = o0;
}

int main(int argc, char *argv[])
{
    const std::vector<int64_t> input_shape = {1, 2};
    double* input_data = new double[2] { 1.0, 1.0 };
    const std::vector<int64_t> output_shape = {1, 2};
    double* output_data = new double[2] { -13.37, -13.37 };

    std::cout << "input_data = " << input_data << std::endl;
    std::cout << "output_data = " << output_data << std::endl;

    TF_Graph* graph;
    TF_Output* graph_inputs;
    TF_Output* graph_outputs;
    int num_graph_inputs = 1;
    int num_graph_outputs = 1;
    TF_Status* status = TF_NewStatus();
    TF_Session* session;
    int device_id = 0;

    initSession(graph, session, graph_inputs, graph_outputs, num_graph_inputs, num_graph_outputs, device_id);

    TF_Tensor* input_tensor;
    TF_Tensor* output_tensor;

    int num_input_elems = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
    size_t input_len = sizeof(double) * num_input_elems;
    char* input_arg = "input tensor";
    //input_tensor = TF_NewTensor(TF_DOUBLE, input_shape.data(), input_shape.size(), (void*) input_data, input_len, fake_tensor_deallocator, input_arg);
    input_tensor = TF_AllocateTensor(TF_DOUBLE, input_shape.data(), input_shape.size(), input_len);
    if ( input_tensor == nullptr )
    {
        std::cerr << "Error: could not allocate input tensor" << std::endl;
        std::cout << "Error: could not allocate input tensor" << std::endl; 
    }

    int num_output_elems = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    size_t output_len = sizeof(double) * num_output_elems;
    char* output_arg = "output tensor";
    //output_tensor = TF_NewTensor(TF_DOUBLE, output_shape.data(), output_shape.size(), (void*) output_data, output_len, fake_tensor_deallocator, output_arg);
    output_tensor = TF_AllocateTensor(TF_DOUBLE, output_shape.data(), output_shape.size(), output_len);
    if ( output_tensor == nullptr )
    {
        std::cerr << "Error: could not allocate output tensor" << std::endl;
        std::cout << "Error: could not allocate output tensor" << std::endl; 
    }

    // set data for allocated tensors
    double* input_res = (double*) TF_TensorData(input_tensor);
    double* output_res = (double*) TF_TensorData(output_tensor);

    input_res[0] = 1.0;
    input_res[1] = 1.0;
    output_res[0] = -42.24;
    output_res[1] = -42.24;

    std::cout << "input_res before = " << input_res << std::endl;
    std::cout << "output_res before = " << output_res << std::endl;

    std::cout << "Starting inference" << std::endl; 
    TF_SessionRun(session, NULL, graph_inputs, &input_tensor, num_graph_inputs, graph_outputs, &output_tensor, num_graph_outputs, NULL, 0, NULL, status);
    if (TF_GetCode(status))
    {
        std::cout << "Tensorflow Run Session Error: " << TF_Message(status) << std::endl;
    }

    std::cout << "(" << input_data[0] << ", " << input_data[1] << ") --> (" << output_data[0] << ", " << output_data[1] << ")" << std::endl;

    input_res = (double*) TF_TensorData(input_tensor);
    output_res = (double*) TF_TensorData(output_tensor);

    std::cout << "(" << input_res[0] << ", " << input_res[1] << ") --> (" << output_res[0] << ", " << output_res[1] << ")" << std::endl;

    std::cout << "input_res after = " << input_res << std::endl;
    std::cout << "output_res after = " << output_res << std::endl;

    std::cout << "Starting inference 2" << std::endl; 
    TF_SessionRun(session, NULL, graph_inputs, &input_tensor, num_graph_inputs, graph_outputs, &output_tensor, num_graph_outputs, NULL, 0, NULL, status);
    if (TF_GetCode(status))
    {
        std::cout << "Tensorflow Run Session Error: " << TF_Message(status) << std::endl;
    }

    std::cout << "(" << input_data[0] << ", " << input_data[1] << ") --> (" << output_data[0] << ", " << output_data[1] << ")" << std::endl;

    input_res = (double*) TF_TensorData(input_tensor);
    output_res = (double*) TF_TensorData(output_tensor);

    std::cout << "(" << input_res[0] << ", " << input_res[1] << ") --> (" << output_res[0] << ", " << output_res[1] << ")" << std::endl;

    std::cout << "input_res after = " << input_res << std::endl;
    std::cout << "output_res after = " << output_res << std::endl;

    return 0;
}