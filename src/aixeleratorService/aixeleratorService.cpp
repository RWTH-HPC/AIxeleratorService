#include "aixeleratorService/aixeleratorService.h"

#include "distributionStrategy/roundRobinDistribution.h"
#include "communicationStrategy/collectiveCommunication.h"

#ifdef WITH_TORCH
#include "inferenceStrategy/torchInference/torchInference.h"
#endif
#ifdef WITH_TENSORFLOW
#include "inferenceStrategy/tensorflowInference/tensorflowInference.h"
#endif
#ifdef WITH_SOL
#include "inferenceStrategy/solInference/solInference.h"
#endif

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
        batchsize_{batchsize} 
{
    registerModel(model_file);
#ifdef WITH_SOL
    if ( framework_ == AIX_SOL )
    {
        /*
        note(fabian): this will become problematic if the user creates multiple AIxeleratorService objects with SOL backend. 
        In this case we only need to initialize once.
        */
        vedaInit(0);
    }
#endif

    distributor_ = std::make_unique<RoundRobinDistribution>();

    communicator_ = std::make_unique<CollectiveCommunication>(input_shape, input_data, output_shape, output_data, distributor_->isGPUController(), *(distributor_->getWorkGroupCommunicator()) );

    int num_devices_total = distributor_->getNumDevicesTotal();
    if (num_devices_total > 0)
    {
        inference_mode_ = AIX_GPU;    

        // inference_mode_ = AIX_HYBRID;
    }
    else
    {
        inference_mode_ = AIX_CPU;
        batchsize_ = std::min<int>(batchsize_, input_shape_[0]); // TODO(fabian): is that the right place to check it?
    }
    
    initInferenceStrategy();
}


AIxeleratorService::~AIxeleratorService()
{
    inferencing_.reset();
    distributor_.reset();

#ifdef WITH_SOL
    if( framework_ == AIX_SOL )
    {
        /*
        note(fabian): this will become problematic if the user creates multiple AIxeleratorService objects with SOL backend. 
        In this case we only need to initialize once.
        */
        vedaExit();
    }
#endif
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

    suffix = model_file.substr(len-4);
    if ( suffix == ".vso" )
    {
        return AIX_SOL;
    }

    return AIX_UNKNOWN;    
}

void AIxeleratorService::registerModel(std::string model_file)
{
    // TODO: only GPU-controller ranks should create unique pointers with inference strategy. This requires to initalize distribution in constructor
    model_file_name_ = model_file;
    framework_ = getAIFrameworkFromModel(model_file_name_);
}

std::unique_ptr<InferenceStrategy> AIxeleratorService::createInferenceStrategy()
{
    std::unique_ptr<InferenceStrategy> strategy;
    switch(framework_)
    {
        case AIX_TORCH:
#ifdef WITH_TORCH
            strategy = std::make_unique<TorchInference>();
#else   
            std::cerr << "Error: AIxeleratorService was not built with Torch backend!" << std::endl;
#endif
            break;
        case AIX_TENSORFLOW:
#ifdef WITH_TENSORFLOW
            strategy = std::make_unique<TensorflowInference>();
#else
            std::cerr << "Error: AIxeleratorService was not built with Tensorflow backend!" << std::endl;
#endif
            break;
        case AIX_SOL:
#ifdef WITH_SOL
            strategy = std::make_unique<SOLInference>();
#else
            std::cerr << "Error: AIxeleratorService was not built with SOL backend!" << std::endl;
#endif
            break;
        case AIX_UNKNOWN:
            std::cerr << "Error: AIxeleratorService does not support format of model file: " << model_file_name_ << std::endl;
            break;
    }

    return std::move(strategy);
}

void AIxeleratorService::initInferenceStrategy()
{
    switch(inference_mode_)
    {
        case AIX_HYBRID:
        {
            double batch_fraction_device = 0.9;
            inferencing_host_ = createInferenceStrategy();
            if(distributor_->isGPUController())
            {
                inferencing_device_ = createInferenceStrategy();
                int device_id = distributor_->getDeviceID();
            }
            break;
        }
        case AIX_GPU:
        {
            if (distributor_->isGPUController())
            {
                inferencing_ = createInferenceStrategy();

                int device_id = distributor_->getDeviceID();
                double* input_data_controller = communicator_->getInputDataController();
                double* output_data_controller = communicator_->getOutputDataController();

                std::vector<int64_t> input_shape_controller = communicator_->getInputShapeController();
                std::vector<int64_t> output_shape_controller = communicator_->getOutputShapeController();

                inferencing_->init(batchsize_, device_id, model_file_name_, input_shape_controller, input_data_controller, output_shape_controller, output_data_controller);
            }
            break;
        }
        case AIX_CPU:
        {
            int device_id = -1;

            inferencing_ = createInferenceStrategy();

            inferencing_->init(batchsize_, device_id, model_file_name_, input_shape_, input_data_, output_shape_, output_data_);
            break;
        }
        default:
        {
            std::cerr << "Error: AIxeleratorService inference mode: " << inference_mode_ << std::endl;
            break;
        }
    }
}

// note(fabian): this function is currently not used
void AIxeleratorService::registerTensors(
    std::vector<int64_t> input_shape, double* input_data,
    std::vector<int64_t> output_shape, double* output_data
){
    input_shape_ = input_shape;
    output_shape_ = output_shape;

    input_data_ = input_data;
    output_data_ = output_data;

    distributor_ = std::make_unique<RoundRobinDistribution>();

    initInferenceStrategy();
}

void AIxeleratorService::inference()
{
    // TODO: add local data copy
    if ( inference_mode_ == AIX_GPU )
    {
        communicator_->gatherInputData();

        if ( distributor_->isGPUController() )
        {
            inferencing_->inference();
        }

        communicator_->scatterOutputData();
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