#include "AIModel.h"
#include "TorchModel.h"
#include <gtest/gtest.h>

const std::string model_filename { "model.pt" };

TEST(AIModel, ConstructTorchModel)
{
    AIxelerator::TorchModel torchmodel(model_filename);
    torchmodel.forward();
}

TEST(AIModel, ConstructFromEnum)
{
    AIxelerator::AIModel aimodel(AIxelerator::AIFramework::pytorch, model_filename);
    aimodel.forward();
}

TEST(AIModel, ConstructFromString)
{
    // 3) Create wrapper class AIModel, 2. constructor
    AIxelerator::AIModel aimodel("pytorch", model_filename);
    aimodel.forward();
}

TEST(AIModel, ConstructFromFilename)
{
    AIxelerator::AIModel aimodel(model_filename);
    aimodel.forward();
}