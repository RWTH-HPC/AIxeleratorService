#include "TorchModel.h"

namespace AIxelerator
{

REGISTER_AIMODEL(TorchModel, "pytorch");

TorchModel::TorchModel(std::string modelFile) 
: AIModelBase(),
options_{torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)}
{
    try {
        model_ = torch::jit::load(modelFile);
        // Unfortunately, there is no nicer way to get the dtype and device from TensorOptions
        model_.to(torch::typeMetaToScalarType(options_.dtype()));
        model_.to(options_.device());
    }
    catch (const c10::Error& e) {
        std::string msg("TorchModel::TorchModel(std::string)\nError loading torch model with error message:\n" + e.msg());
        throw std::runtime_error(msg);
    }

    input_ = torch::ones({2,2}, options_);

}

void TorchModel::forward()
{
    std::cout<< "TorchModel::forward()" << std::endl;
    output_ = model_.forward(std::vector<torch::IValue>{input_}).toTensor();
    std::cout << output_ << std::endl;
}

} // namespace AIxelerator