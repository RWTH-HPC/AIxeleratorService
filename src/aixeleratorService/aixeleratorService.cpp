#include "aixeleratorService.h"

#include "distributionStrategy/roundRobinDistribution.h"

#include "inferenceStrategy/torchInference/torchInference.h"
#include "inferenceStrategy/tensorflowInference/tensorflowInference.h"

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
    int num_devices_total = distributor_->getNumDevicesTotal();
    if (num_devices_total > 0)
    {
        inference_mode_ = AIX_GPU;    
    }
    else
    {
        inference_mode_ = AIX_CPU;
        batchsize_ = std::min<int>(batchsize_, input_shape_[0]); // TODO(fabian): is that the right place to check it?
    }

    registerModel(model_file);
    if( framework_ == AIX_TENSORFLOW && inference_mode_ == AIX_CPU)
    {
        std::cerr << "Error: AIxeleratorService does not support pure CPU inference with Tensorflow (yet)." << std::endl;
    }
    
    initInferenceStrategy();
}

AIFramework getAIFrameworkFromModel(std::string model_file)
{
    // note(fabian): string.ends_with() requires C++20
    int len = model_file.length();
    std::string suffix = model_file.substr(len-3);
    //if ( model_file.ends_with(std::string_view(".pt")) )
    if ( suffix == ".pt" )
    {
        return AIX_TORCH;
    }
    
    //if ( model_file.ends_with(std::string_view(".pb")) || model_file.ends_with(std::string_view(".tf")) )
    if ( (suffix == ".pb") || (suffix == ".tf") )
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
    switch(inference_mode_)
    {
        case AIX_GPU:
            if (distributor_->isGPUController())
            {
                int device_id = distributor_->getDeviceID();
                double* input_data_controller = distributor_->getInputDataController();
                double* output_data_controller = distributor_->getOutputDataController();

                std::vector<int64_t> input_shape_controller = distributor_->getInputShapeController();
                std::vector<int64_t> output_shape_controller = distributor_->getOutputShapeController();

                inferencing_->init(batchsize_, device_id, model_file_name_, input_shape_controller, input_data_controller, output_shape_controller, output_data_controller);
            }
            break;
        case AIX_CPU:
            int device_id = -1;
            inferencing_->init(batchsize_, device_id, model_file_name_, input_shape_, input_data_, output_shape_, output_data_);
            break;
        default:
            std::cerr << "Error: AIxeleratorService inference mode: " << inference_mode_ << std::endl;
            break;
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
    if ( inference_mode_ == AIX_GPU )
    {
        distributor_->gatherInputData();

        if ( distributor_->isGPUController() )
        {
            inferencing_->inference();
        }

        distributor_->scatterOutputData();
    }
    else if ( inference_mode_ == AIX_CPU )
    {
        inferencing_->inference();   
    }
    else
    {
        std::cerr << "Error: AIxeleratorService does not support inference_mode " << inference_mode_ << std::endl;
    }

    

    // TODO: add local data copy
}