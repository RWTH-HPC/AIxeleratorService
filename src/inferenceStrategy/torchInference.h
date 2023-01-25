#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_TORCHINFERENCE_H_
#define AIXELERATORSERVICE_INFERENCESTRATEGY_TORCHINFERENCE_H_

#include "inferenceStrategy/inferenceStrategy.h"
#include <torch/script.h>

class TorchInference : public InferenceStrategy
{
    public:
        TorchInference() = default;
        ~TorchInference() = default;

        void init(
            int batchsize, 
            int device_id, 
            std::string model_file_name, 
            const std::vector<int64_t>& input_shape, double* inputData, 
            const std::vector<int64_t>& output_shape, double* outputData
        );
        void inference() override;

    private:
        std::string model_file_name_;
        torch::jit::script::Module torch_model_;
        int batchsize_;
        int device_id_;
        torch::Tensor input_;
        torch::Tensor input_batch_;
        torch::Tensor input_gpu_;
        torch::Tensor output_;
        torch::Tensor output_batch_;
        torch::Tensor output_gpu_;
};


#endif