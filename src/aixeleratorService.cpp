#include "aixeleratorService.h"

#include "distributionStrategy/roundRobinDistribution.h"

#include "inferenceStrategy/torchInference.h"
#include "inferenceStrategy/tensorflowInference.h"

#include <numeric>
#include <iostream>

AIxeleratorService::AIxeleratorService(
    std::string model_file,
    std::vector<int64_t> input_shape, double* input_data,
    std::vector<int64_t> output_shape, double* output_data,
    int batchsize
)   :   model_file_name_{model_file}, 
        input_shape_{input_shape}, input_data_{input_data}, 
        output_shape_{output_shape}, output_data_{output_data}, 
        batchsize_{batchsize}, 
        distributor_{std::make_unique<RoundRobinDistribution>(input_shape, input_data, output_shape, output_data)}
{
    registerModel(model_file);
    
    initInferenceStrategy();
}

AIFramework getAIFrameworkFromModel(std::string model_file)
{
    // note(fabian): string.ends_with() requires C++20
    if ( model_file.ends_with(std::string_view(".pt")) )
    {
        return AIX_TORCH;
    }
    
    if ( model_file.ends_with(std::string_view(".pb")) || model_file.ends_with(std::string_view(".tf")) )
    {
        return AIX_TENSORFLOW;
    }
    else
    {
        return AIX_UNKNOWN;
    }    
}

void AIxeleratorService::registerModel(std::string model_file)
{
    // TODO: only GPU-controller ranks should create unique pointers with inference strategy. This requires to initalize distribution in constructor
    model_file_name_ = model_file;
    framework_ = getAIFrameworkFromModel(model_file_name_);

    switch(framework_)
    {
        case AIX_TORCH:
            inferencing_ = std::make_unique<TorchInference>();
            break;
        case AIX_TENSORFLOW:
            inferencing_ = std::make_unique<TensorflowInference>();
            break;
        case AIX_UNKNOWN:
            std::cerr << "Error: AIxeleratorService does not support format of model file: " << model_file_name_ << std::endl;
            break;
    }
}

void AIxeleratorService::initInferenceStrategy()
{
    if (distributor_->isGPUController())
    {
        int device_id = distributor_->getDeviceID();
        double* input_data_controller = distributor_->getInputDataController();
        double* output_data_controller = distributor_->getOutputDataController();

        std::vector<int64_t> input_shape_controller = distributor_->getInputShapeController();
        std::vector<int64_t> output_shape_controller = distributor_->getOutputShapeController();

        inferencing_->init(batchsize_, device_id, model_file_name_, input_shape_controller, input_data_controller, output_shape_controller, output_data_controller);
    }
}

void AIxeleratorService::registerTensors(
    std::vector<int64_t> input_shape, double* input_data,
    std::vector<int64_t> output_shape, double* output_data
){
    input_shape_ = input_shape;
    output_shape_ = output_shape;

    input_data_ = input_data;
    output_data_ = output_data;

    distributor_ = std::make_unique<RoundRobinDistribution>(input_shape_, input_data, output_shape_, output_data);

    initInferenceStrategy();
}

void AIxeleratorService::inference()
{
    // TODO: add local data copy

    distributor_->gatherInputData();

    if ( distributor_->isGPUController() )
    {
        inferencing_->inference();
    }

    distributor_->scatterOutputData();

    // TODO: add local data copy
}