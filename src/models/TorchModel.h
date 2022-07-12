#include "AIModel.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <string>

class TorchModel : public AIModel<torch::jit::Module>
{
    public:
        TorchModel() = delete;
        TorchModel(std::string modelFile);
        ~TorchModel();
        void forward(double* inputTensorData, double* outputTensorData);
};