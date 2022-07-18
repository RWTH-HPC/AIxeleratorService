#include "AIModel.h"
#include "TorchModel.h"

int main()
{
    std::string model_filename{"model.pt"};
    // 1) Create model directly
    AIxelerator::TorchModel torchmodel(model_filename);
    torchmodel.forward();
        
    // 2) Create wrapper class AIModel, 1. constructor    
    AIxelerator::AIModel aimodel1(AIxelerator::AIFramework::pytorch, model_filename);
    aimodel1.forward();

    // 3) Create wrapper class AIModel, 2. constructor 
    AIxelerator::AIModel aimodel2("pytorch", model_filename);
    aimodel2.forward();

    // 4) Create wrapper class AIModel, 3. constructor 
    AIxelerator::AIModel aimodel3(model_filename);
    aimodel3.forward();

    return 0;
}