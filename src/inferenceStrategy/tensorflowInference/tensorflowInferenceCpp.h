#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_TENSORFLOWINFERENCECPP_H_
#define AIXELERATORSERVICE_INFERENCESTRATEGY_TENSORFLOWINFERENCECPP_H_

#include "inferenceStrategy/inferenceStrategy.h"

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_slice.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/core/public/session.h>

#include <vector>
#include <string>

template<typename T>
class TensorflowInferenceCpp : public InferenceStrategy<T>
{
    public:
        TensorflowInferenceCpp();
        ~TensorflowInferenceCpp();

        void init(
            int batchsize, 
            int device_id, 
            std::string model_file_name, 
            std::vector<int64_t>& input_shape, T* input_data, 
            std::vector<int64_t>& output_shape, T* output_data
        ) override;
        void inference() override;

    private:
    tensorflow::SavedModelBundle tf_model_bundle_;
    tensorflow::SessionOptions session_options_;
    tensorflow::RunOptions run_options_;

    void initSession();

    std::string model_file_name_;
    int device_id_;
    int batchsize_;
    int num_batches_;
    int num_tensors_;
    int size_remaining_;

    T* app_input_;
    T* app_output_;

    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_;
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_remainder_;
    std::vector<tensorflow::Tensor> output_tensor_batch_;
    std::vector<tensorflow::Tensor> output_tensor_remainder_;
    size_t num_elems_input_batch_;
    size_t size_input_batch_;
    size_t num_elems_input_remainder_;
    size_t size_input_remainder_;
    size_t num_elems_output_batch_;
    size_t size_output_batch_;
    size_t num_elems_output_remainder_;
    size_t size_output_remainder_;

    std::vector<std::string> input_tensor_names_;
    std::vector<std::string> output_tensor_names_;
    std::vector<int64_t> input_tensor_offsets_;
    std::vector<int64_t> output_tensor_offsets_;

    void initTensors(
            std::vector<int64_t>& input_shape, T* input_data, 
            std::vector<int64_t>& output_shape, T* output_data
    );
};

#endif