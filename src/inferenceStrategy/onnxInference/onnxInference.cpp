//
// Created by co007276 on 8/4/23.
//
#include "inferenceStrategy/onnxInference/onnxInference.h"

#include <iostream>
#include <algorithm>  // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>


// Taken from: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/model-explorer/model-explorer.cpp
template <typename T>
std::vector<std::vector<Ort::Value>> vec_to_tensor_batch(T* data, const std::vector<std::int64_t>& shape,
                                                         int64_t batch_size, int num_batches) {
    std::vector<std::vector<Ort::Value>> batch_input_tensors(num_batches);
    long num_ele;
    int64_t cur_batch_size;
    int data_size = 1;
    int64_t batch_dim = shape[0];
    std::vector<std::int64_t> new_shape = shape;


    std::for_each(shape.begin(), shape.end(), [&] (int n) {
        data_size *= n;
    });

    for (int64_t i = 0; i < num_batches * batch_size; i += batch_size) {
        Ort::MemoryInfo mem_info =
                Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                           OrtMemType::OrtMemTypeDefault);
        // Set cur_batch_size in case of remainder elements
        cur_batch_size = std::min(batch_dim - i, batch_size);
        new_shape.at(0) = std::ceil(cur_batch_size);

        // Calculate number of elements for current batch
        num_ele = cur_batch_size  * (data_size / batch_dim);
        auto cur_data_pointer = data + i * (data_size / batch_dim);
        batch_input_tensors[i / batch_size].push_back(Ort::Value::CreateTensor<T>(mem_info,
                                                                                  cur_data_pointer,
                                                                                  num_ele, &new_shape[0],
                                                                                  new_shape.size()));
    }
    return batch_input_tensors;
}

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t>& v) {
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}


template<typename T>
void ONNXInference<T>::init(
        int batchsize,
        int device_id,
        std::string model_file_name,
        std::vector<int64_t>& input_shape, T* inputData,
        std::vector<int64_t>& output_shape, T* outputData
){
    device_id_ = device_id;

    batchsize_ = batchsize;
    if (batchsize_ < 1)
    {
        std::cerr << "Error in init ONNXInference: batchsize should not be zero or negative!" << std::endl;
    }

    // onnxruntime setup

    // input data setup
    // print name/shape of inputs
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::int64_t> input_shapes;
    std::cout << "Input Node Name/Shape (" << input_names_.size() << "):" << std::endl;
    for (std::size_t i = 0; i < session_.GetInputCount(); i++) {
        input_names_.emplace_back(session_.GetInputNameAllocated(i, allocator).get());
        input_shapes = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "\t" << input_names_.at(i) << " : " << print_shape(input_shapes) << std::endl;
    }
    // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
    for (auto& s : input_shapes) {
        if (s < 0) {
            s = 1;
        }
    }

    // print name/shape of outputs
    std::cout << "Output Node Name/Shape (" << output_names_.size() << "):" << std::endl;
    for (std::size_t i = 0; i < session_.GetOutputCount(); i++) {
        output_names_.emplace_back(session_.GetOutputNameAllocated(i, allocator).get());
        auto output_shapes = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "\t" << output_names_.at(i) << " : " << print_shape(output_shapes) << std::endl;
    }


    batch_dim_ = input_shape[0];
    num_batches_ = std::ceil(batch_dim_ / (long double)batchsize_);
    std::cout << "Creating input batches" << std::endl;
    input_ = vec_to_tensor_batch<T>(inputData, input_shape, batchsize_, num_batches_);
    output_ = outputData;
}

template<typename T>
void ONNXInference<T>::inference()
{
    std::vector<const char*> input_names_char(input_names_.size(), nullptr);
    std::transform(std::begin(input_names_), std::end(input_names_), std::begin(input_names_char),
                   [&](const std::string& str) { return str.c_str(); });

    std::vector<const char*> output_names_char(output_names_.size(), nullptr);
    std::transform(std::begin(output_names_), std::end(output_names_), std::begin(output_names_char),
                   [&](const std::string& str) { return str.c_str(); });

    std::cout << "Running model..." << std::endl;
    int64_t ele_count = 0;
    for (int64_t i = 0; i < num_batches_; i++) {
        std::cout << "\nProcessing batch #" << i << std::endl;
        auto batch_output_tensor = session_.Run(Ort::RunOptions{nullptr},
                                                input_names_char.data(),
                                                input_[i].data(),
                                                input_names_char.size(),
                                                output_names_char.data(),
                                                output_names_char.size());
        //output_.insert(output_.end(), batch_output_tensor.begin(), batch_output_tensor.end());
        T* val_from_tensor = batch_output_tensor[0].GetTensorMutableData<T>();
        auto outputInfo = batch_output_tensor[0].GetTensorTypeAndShapeInfo();
        int64_t cur_ele_count = 1;
        std::cout << "GetElementType: " << outputInfo.GetElementType() << "\n";
        std::cout << "Dimensions of the output: " << outputInfo.GetShape().size() << "\n";
        std::cout << "Shape of the output: ";
        for (unsigned int shapeI = 0; shapeI < outputInfo.GetShape().size(); shapeI++) {
            std::cout << outputInfo.GetShape()[shapeI] << ", ";
            cur_ele_count *= outputInfo.GetShape()[shapeI];
        }
        std::cout << "\n";
        //cur_ele_count = std::min(batch_dim_ - (i * batchsize_), batchsize_); // TODO calculates current batch size, not element count
        //cur_ele_count = outputInfo.GetShape()[0];
        std::copy(val_from_tensor, val_from_tensor + cur_ele_count, output_ + ele_count);
        ele_count = cur_ele_count;
    }
}


template class ONNXInference<float>;
template class ONNXInference<double>;