#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_TENSORFLOWINFERENCE_H_
#define AIXELERATORSERVICE_INFERENCESTRATEGY_TENSORFLOWINFERENCE_H_

#include "inferenceStrategy/inferenceStrategy.h"

#include <tensorflow/core/protobuf/config.pb.h>
#include <tensorflow/c/c_api.h>
#include <tensorflow/c/c_api_experimental.h>

#include <vector>
#include <string>

template<typename T>
class TensorflowInference : public InferenceStrategy<T>
{
    public:
        TensorflowInference();
        ~TensorflowInference();

        void init(
            int batchsize, 
            int device_id, 
            std::string model_file_name, 
            std::vector<int64_t>& input_shape, T* input_data, 
            std::vector<int64_t>& output_shape, T* output_data
        ) override;
        void inference() override;

    private:
        std::string model_file_name_;
        TF_Graph* graph_;
        int num_graph_inputs_;
        int num_graph_outputs_;
        TF_Output* graph_outputs_;
        TF_Output* graph_inputs_;

        int batchsize_;
        int device_id_;
        int num_batches_;
        int size_remaining_;

        TF_Status* status_;
        TF_SessionOptions* session_opts_;
        TF_Session* session_;
        TF_Buffer* run_options_;
        TF_Buffer* run_metadata_;

        TF_Tensor* input_batch_;
        size_t input_batch_len_;
        TF_Tensor* output_batch_;
        size_t output_batch_len_;
        TF_Tensor* input_remainder_;
        size_t input_remainder_len_;
        TF_Tensor* output_remainder_;
        size_t output_remainder_len_;

        std::vector<int> tensor_offsets_in_;
        std::vector<int> tensor_offsets_out_;

        T* app_input_;
        T* app_output_;

        void initSession();
        void initTensors(
            std::vector<int64_t>& input_shape, T* input_data, 
            std::vector<int64_t>& output_shape, T* output_data
        );

        int debug_count_;
};

#endif