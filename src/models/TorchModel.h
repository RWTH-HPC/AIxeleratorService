#pragma once

#include "AIModel.h"
#include "torch/script.h"
#include <string>

namespace AIxelerator {

class TorchModel : public AIModelBase {
    torch::jit::Module model_ {};
    torch::TensorOptions options_ {};
    torch::Tensor input_ {};
    torch::Tensor output_ {};

public:
    TorchModel() = delete;
    explicit TorchModel(std::string modelFile);
    ~TorchModel() override = default;
    void forward() override;
};

} // namespace AIxelerator