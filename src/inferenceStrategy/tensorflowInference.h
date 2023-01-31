#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_TENSORFLOWINFERENCE_H_
#define AIXELERATORSERVICE_INFERENCESTRATEGY_TENSORFLOWINFERENCE_H_

#include "inferenceStrategy/inferenceStrategy.h"

#include <tensorflow/c/c_api.h>

#include <vector>
#include <string>

class TensorflowInference : public InferenceStrategy
{
    public:
        TensorflowInference() = default;
        ~TensorflowInference() = default;

        void init(
            int batchsize, 
            int device_id, 
            std::string model_file_name, 
            std::vector<int64_t>& input_shape, double* input_data, 
            std::vector<int64_t>& output_shape, double* output_data
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

        TF_Tensor* input_;
        TF_Tensor* output_;
        TF_Tensor* input_batch_;
        TF_Tensor* output_batch_;
        TF_Tensor* input_remainder_;
        TF_Tensor* output_remainder_;

        std::vector<TF_Tensor*> input_tensors_;
        std::vector<TF_Tensor*> output_tensors_;
        std::vector<int> tensor_offsets_;
        std::vector<int> tensor_sizes_;

        double* app_output_;

        void initSession();
        void initTensors(
            std::vector<int64_t>& input_shape, double* input_data, 
            std::vector<int64_t>& output_shape, double* output_data
        );
};

#endif