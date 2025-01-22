#include "aixeleratorService/aixeleratorService.h"

#include "distributionStrategy/roundRobinDistribution.h"
#include "communicationStrategy/collectiveCommunication.h"
#include "communicationStrategy/nonBlockingPtoPCommunication.h"

#ifdef WITH_TORCH
#include "inferenceStrategy/torchInference/torchInference.h"
#endif
#ifdef WITH_TENSORFLOW
#include "inferenceStrategy/tensorflowInference/tensorflowInference.h"
#endif
#ifdef WITH_ONNX
#include "inferenceStrategy/onnxInference/onnxInference.h"
#endif
#ifdef WITH_SOL
#include "inferenceStrategy/solInference/solInference.h"
#endif

#ifdef SCOREP
#include <scorep/SCOREP_User.h>

SCOREP_USER_REGION_DEFINE( gatherHandle )
SCOREP_USER_REGION_DEFINE( deviceInferenceHandle )
SCOREP_USER_REGION_DEFINE( hostInferenceHandle )
SCOREP_USER_REGION_DEFINE( scatterHandle )
#endif

#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>

template<typename T>
AIxeleratorService<T>::AIxeleratorService(
    std::string model_file,
    std::vector<int64_t> input_shape, T* input_data,
    std::vector<int64_t> output_shape, T* output_data,
    int batchsize, MPI_Comm app_comm
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

    distributor_ = std::make_unique<RoundRobinDistribution>(app_comm);
    std::pair<int64_t, int64_t> best_batchsizes(-1, -1);

    int num_devices_total = distributor_->getNumDevicesTotal();
    if (num_devices_total > 0)
    {
        
        communicator_ = std::make_unique<CollectiveCommunication<T>>( 
            input_shape, input_data, 
            output_shape, output_data, 
            distributor_->isGPUController(), *(distributor_->getWorkGroupCommunicator()) 
        );

        if (distributor_->isGPUController() )
        {
            best_batchsizes = {0, input_shape[0]};
        }
        MPI_Bcast(&best_batchsizes.first, 1, MPI_INT64_T, 0, *(distributor_->getWorkGroupCommunicator()) );
        MPI_Bcast(&best_batchsizes.second, 1, MPI_INT64_T, 0, *(distributor_->getWorkGroupCommunicator()) );
    }
    else
    {
        communicator_ = nullptr;
        batchsize_ = std::min<int>(batchsize_, input_shape_[0]); // TODO(fabian): is that the right place to check it?

        // in the absence of devices, all samples should be inferred on the host (CPU)
        best_batchsizes = {input_shape_[0], 0};
    }
    
    MPI_Comm_rank(*distributor_->getWorkGroupCommunicator(), &my_rank_);

    initInferenceStrategy(best_batchsizes);
}

template<typename T>
AIxeleratorService<T>::~AIxeleratorService()
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
    std::cout << "AIxeleratorService: suffix = " << suffix << "\n";
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

    if ( suffix == ".ort" )
    {
        return AIX_ONNX;
    }

    suffix = model_file.substr(len-5);
    if ( suffix == ".onnx" )
    {
        return AIX_ONNX;
    }

    return AIX_UNKNOWN;    
}

template<typename T>
void AIxeleratorService<T>::registerModel(std::string model_file)
{
    // TODO: only GPU-controller ranks should create unique pointers with inference strategy. This requires to initalize distribution in constructor
    model_file_name_ = model_file;
    framework_ = getAIFrameworkFromModel(model_file_name_);
}

template<typename T>
std::unique_ptr<InferenceStrategy<T>> AIxeleratorService<T>::createInferenceStrategy()
{
    std::unique_ptr<InferenceStrategy<T>> strategy;
    switch(framework_)
    {
        case AIX_TORCH:
#ifdef WITH_TORCH
            strategy = std::make_unique<TorchInference<T>>();
#else   
            std::cerr << "Error: AIxeleratorService was not built with Torch backend!" << std::endl;
#endif
            break;
        case AIX_TENSORFLOW:
#ifdef WITH_TENSORFLOW
            strategy = std::make_unique<TensorflowInference<T>>();
            //strategy = std::make_unique<TensorflowInferenceCpp<T>>();
#else
            std::cerr << "Error: AIxeleratorService was not built with Tensorflow backend!" << std::endl;
#endif
            break;
        case AIX_ONNX:
#ifdef WITH_ONNX
            strategy = std::make_unique<ONNXInference<T>>(model_file_name_);
#else
            std::cerr << "Error: AIxeleratorService was not built with ONNX runtime backend!" << std::endl;
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


template<typename T>
void AIxeleratorService<T>::initInferenceStrategy(std::pair<int64_t, int64_t> best_batchsizes)
{
    // determine batch dimension for host & device
    int64_t batch_host = best_batchsizes.first;
    int64_t batch_device = best_batchsizes.second;

    // TODO (fabian): remove this workaround as soon as we implement asynchronous device inference
    if( distributor_->isGPUController() )
    {
        batch_host = 0;
    }

    // determine input data shape for host and device
    input_shape_host_ = input_shape_;
    input_shape_host_[0] = batch_host;
    input_shape_device_ = input_shape_;
    //input_shape_device_[0] = batch_device;
    input_shape_device_[0] = input_shape_[0] - batch_host;

    output_shape_host_ = output_shape_;
    output_shape_host_[0] = batch_host;
    output_shape_device_ = output_shape_;
    //output_shape_device_[0] = batch_device;
    output_shape_device_[0] = output_shape_[0] - batch_host;

    // setup data pointers accordingly
    input_data_host_ = &input_data_[0];
    int64_t num_input_elem_per_batch = std::accumulate(std::next(input_shape_.begin(), 1), input_shape_.end(), 1, std::multiplies<int>());
    input_data_device_ = &input_data_[batch_host * num_input_elem_per_batch];

    output_data_host_ = &output_data_[0];
    int64_t num_output_elem_per_batch = std::accumulate(std::next(output_shape_.begin(), 1), output_shape_.end(), 1, std::multiplies<int>());
    output_data_device_ = &output_data_[batch_host * num_output_elem_per_batch];

    // create communicator only for the device partition of input_data
    communicator_.reset();
    if (distributor_->getNumDevicesTotal() > 0)
    {
        
        communicator_ = std::make_unique<CollectiveCommunication<T>>(
            input_shape_device_, input_data_device_, 
            output_shape_device_, output_data_device_, 
            distributor_->isGPUController(), 
            *(distributor_->getWorkGroupCommunicator())
        );
        
    }

    inferencing_host_ = createInferenceStrategy();
    if (batch_host > 0){
        inferencing_host_->init(batch_host, -1, model_file_name_, input_shape_host_, input_data_host_, output_shape_host_, output_data_host_);
    }
    if(distributor_->isGPUController())
    {
        inferencing_device_ = createInferenceStrategy();
        int device_id = distributor_->getDeviceID();

        T* input_data_controller = communicator_->getInputDataController();
        T* output_data_controller = communicator_->getOutputDataController();

        std::vector<int64_t> input_shape_controller = communicator_->getInputShapeController();
        std::vector<int64_t> output_shape_controller = communicator_->getOutputShapeController();

        if (batch_device > 0)
        {
            inferencing_device_->init(batchsize_, device_id, model_file_name_, input_shape_controller, input_data_controller, output_shape_controller, output_data_controller);
        }
    }

}

template<typename T>
void AIxeleratorService<T>::inference()
{

#ifdef SCOREP
    SCOREP_USER_REGION( "inference", SCOREP_USER_REGION_TYPE_FUNCTION )
#endif

    if(my_rank_ == 0)
        std::cout << "AIxeleratorService: gathering input data" << std::endl;
#ifdef SCOREP
    SCOREP_USER_REGION_BEGIN( gatherHandle, "gatherInputData", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif
    if( communicator_ )
    {
        communicator_->gatherInputData();
    }
#ifdef SCOREP
    SCOREP_USER_REGION_END( gatherHandle )
#endif

    if(my_rank_ == 0)
        std::cout << "AIxeleratorService: inference on device" << std::endl;
#ifdef SCOREP
    SCOREP_USER_REGION_BEGIN( deviceInferenceHandle, "inferenceDevice", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif
    if ( distributor_->isGPUController() && input_shape_device_[0] > 0)
    {
        inferencing_device_->inference();
    }
#ifdef SCOREP
    SCOREP_USER_REGION_END( deviceInferenceHandle )
#endif
    if(my_rank_ == 0)
        std::cout << "AIxeleratorService: inference on host" << std::endl;
#ifdef SCOREP
    SCOREP_USER_REGION_BEGIN( hostInferenceHandle, "inferenceHost", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif
    // TODO (fabian): enable host inference also for GPU controllers as soon as we implement asynchronous device inference
    if( input_shape_host_[0] > 0 && !distributor_->isGPUController())
    {
        inferencing_host_->inference();
    }
#ifdef SCOREP
    SCOREP_USER_REGION_END( hostInferenceHandle )
#endif
    if(my_rank_ == 0)
        std::cout << "AIxeleratorService: scattering output data" << std::endl;
#ifdef SCOREP
    SCOREP_USER_REGION_BEGIN( scatterHandle, "scatterOutputData", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif
    if( communicator_ )
    {
        communicator_->scatterOutputData();
    }
#ifdef SCOREP
    SCOREP_USER_REGION_END( scatterHandle )
#endif
}

template class AIxeleratorService<float>;
template class AIxeleratorService<double>;