#include "inferenceStrategy/tensorflowInference/tensorflowInferenceCpp.h"

//#include <tensorflow/core/framework/types.pb.h>

#include <iostream>
#include <numeric>
#include <functional>

template<typename T>
TensorflowInferenceCpp<T>::TensorflowInferenceCpp()
{

}

template<typename T>
TensorflowInferenceCpp<T>::~TensorflowInferenceCpp()
{

}

template<typename T>
void TensorflowInferenceCpp<T>::initSession()
{
    auto config = session_options_.config;
    auto device_count = config.mutable_device_count();
    if ( device_id_ > -1)
    {
        device_count->insert({"GPU", 1});

        auto gpu_options = config.mutable_gpu_options();
        gpu_options->set_allow_growth(true);
        gpu_options->set_per_process_gpu_memory_fraction(1.0);
        gpu_options->set_visible_device_list(std::to_string(device_id_));
    }
    else{
        device_count->insert({"CPU", 1});
        device_count->insert({"GPU", 0});

        config.set_intra_op_parallelism_threads(0);
        config.set_inter_op_parallelism_threads(0);
    }

    // init session
    // note: check if {tensorflow::kSavedModelTagServe} == {"serve"} ??
    auto status = tensorflow::LoadSavedModel(session_options_, run_options_, model_file_name_, {"serve"}, &tf_model_bundle_);
    if ( status.ok() )
    {
        std::cout << "TensorflowInferenceCpp: saved model loaded successfully!" << std::endl;
    }
    else 
    {
        std::cout << "TensorflowInferenceCpp: Error: could not load saved model from " << model_file_name_ << std::endl;
    }
}

template<typename T>
tensorflow::DataType getTypeFromTemplate()
{
    if (std::is_same<T, float>::value)
    {
        return tensorflow::DT_FLOAT;  
    }

    if (std::is_same<T, double>::value)
    {
        return tensorflow::DT_DOUBLE;
    }

    std::cerr << "ERROR: Could not get tensorflow::DataType from template parameter!" << std::endl;
    // TODO(fabian): find a nicer way to handle error. Maybe use std::optional as return type?
    return static_cast<tensorflow::DataType>(-1);
}

template<typename T>
void TensorflowInferenceCpp<T>::initTensors(
    std::vector<int64_t>& input_shape, T* input_data, 
    std::vector<int64_t>& output_shape, T* output_data
)
{
    // print information about the model
    auto sig_map = tf_model_bundle_.GetSignatures();
    auto model_def = sig_map.at("serving_default");

    std::cout << "Model Signature" << std::endl;
    for (auto const& p: sig_map)
    {
        std::cout << "key: " << p.first << std::endl;
    }

    input_tensor_names_.clear();
    std::cout << "Model Input Nodes" << std::endl;
    for (auto const& p: model_def.inputs())
    {
        std::cout << "key: " << p.first << " value: " << p.second.name() << std::endl;
        std::string input_name = model_def.inputs().at(p.first).name();
        input_tensor_names_.push_back(input_name);
    }

    output_tensor_names_.clear();
    std::cout << "Model Output Nodes" << std::endl;
    for (auto const& p: model_def.outputs())
    {
        std::cout << "key: " << p.first << " value: " << p.second.name() << std::endl;
        std::string output_name = model_def.outputs().at(p.first).name();
        output_tensor_names_.push_back(output_name);
    }


    // allocate tensors
    int batch_dim = input_shape[0];
    num_batches_ = batch_dim / batchsize_;
    size_remaining_ = batch_dim % batchsize_;
    int num_tensors = size_remaining_ > 0 ? num_batches_+1 : num_batches_;

    int num_input_elems_per_sample = std::accumulate(std::next(input_shape.begin()), input_shape.end(), 1, std::multiplies<int>());
    int num_output_elems_per_sample = std::accumulate(std::next(output_shape.begin()), output_shape.end(), 1, std::multiplies<int>());
    for(int i = 0; i < num_tensors_; i++)
    {
        input_tensor_offsets_[i] = i*batchsize_*num_input_elems_per_sample;
        output_tensor_offsets_[i] = i*batchsize_*num_output_elems_per_sample;
    }

    // create input tensors
    tensorflow::DataType dtype = getTypeFromTemplate<T>();
    std::vector<int64_t> input_shape_batch = input_shape;
    input_shape_batch[0] = batchsize_;
    tensorflow::TensorShape shape(input_shape_batch);
    tensorflow::Tensor input_tensor_batch(dtype, shape);
    inputs_ = { {input_tensor_names_[0], input_tensor_batch} };
    num_elems_input_batch_ = input_tensor_batch.NumElements();
    size_input_batch_ = num_elems_input_batch_ * sizeof(dtype);

    if ( size_remaining_ > 0 )
    {
        std::vector<int64_t> input_shape_remainder = input_shape;
        input_shape_remainder[0] = size_remaining_;
        tensorflow::TensorShape shape(input_shape_remainder);
        tensorflow::Tensor input_tensor_remainder(dtype, shape);
        inputs_remainder_ = { {input_tensor_names_[0], input_tensor_remainder} };
        num_elems_input_remainder_ = input_tensor_remainder.NumElements();
        size_input_remainder_ = num_elems_input_remainder_ * sizeof(dtype);
    }

    // create output tensors
    std::vector<int64_t> output_shape_batch = output_shape;
    output_shape_batch[0] = batchsize_;
    tensorflow::TensorShape shape2(output_shape_batch);
    output_tensor_batch_ = { tensorflow::Tensor(dtype, shape2) };
    num_elems_output_batch_ = output_tensor_batch_[0].NumElements();
    size_output_batch_ = num_elems_output_batch_ * sizeof(dtype);

    if ( size_remaining_ > 0 )
    {
        std::vector<int64_t> output_shape_remainder = output_shape;
        output_shape_remainder[0] = size_remaining_;
        tensorflow::TensorShape shape(output_shape_remainder);
        output_tensor_remainder_ = { tensorflow::Tensor(dtype, shape) };
        num_elems_output_remainder_ = output_tensor_remainder_[0].NumElements();
        size_output_remainder_ = num_elems_output_remainder_ * sizeof(dtype);
    }
}

template<typename T>
void TensorflowInferenceCpp<T>::init(
    int batchsize, 
    int device_id, 
    std::string model_file_name, 
    std::vector<int64_t>& input_shape, T* input_data, 
    std::vector<int64_t>& output_shape, T* output_data
){
    device_id_ = device_id;
    model_file_name_ = model_file_name;
    batchsize_ = batchsize;

    app_input_ = input_data;
    app_output_ = output_data;

    initSession();
    initTensors(input_shape, input_data, output_shape, output_data);

}

template<typename T>
void TensorflowInferenceCpp<T>::inference()
{
    std::string output_name = output_tensor_names_[0];

    for(int i = 0; i < num_batches_; i++)
    {
        // copy data into the input tensor's internal buffer
        tensorflow::Tensor& input_tensor = inputs_[0].second;
        std::memcpy(input_tensor.data(), &(app_input_[input_tensor_offsets_[i]]), size_input_batch_);

        auto status = tf_model_bundle_.GetSession()->Run(inputs_, {output_name}, {}, &output_tensor_batch_);
        if (!status.ok())
        {
            std::cout << "TensorflowInferenceCpp: Error in inference() call batch index = " << i << "!" << std::endl;
        }

        std::memcpy(&(app_output_[output_tensor_offsets_[i]]), output_tensor_batch_.data(), size_output_batch_);
    }

    if ( size_remaining_ > 0 )
    {
        tensorflow::Tensor& input_tensor_remainder = inputs_remainder_[0].second;
        std::memcpy(input_tensor_remainder.data(), &(app_input_[input_tensor_offsets_[num_batches_]]), size_input_remainder_);

        auto status = tf_model_bundle_.GetSession()->Run(inputs_remainder_, {output_name}, {}, &output_tensor_remainder_);
        if (!status.ok())
        {
            std::cout << "TensorflowInferenceCpp: Error in inference() call remainder!" << std::endl;
        }

        std::memcpy(&(app_output_[output_tensor_offsets_[num_batches_]]), output_tensor_remainder_.data(), size_output_remainder_);
    }
}

template class TensorflowInferenceCpp<float>;
template class TensorflowInferenceCpp<double>;
