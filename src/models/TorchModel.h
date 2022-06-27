#include "AIModel.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <string>

class TorchModel : public AIModel
{
    public:
        TorchModel() = delete;
        TorchModel(std::string modelFile);
        ~TorchModel();
};