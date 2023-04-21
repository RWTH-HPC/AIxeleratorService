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
#ifdef WITH_SOL
#include "inferenceStrategy/solInference/solInference.h"
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
    std::pair<int64_t, int64_t> best_batchsizes(-1, -1);

    int num_devices_total = distributor_->getNumDevicesTotal();
    if (num_devices_total > 0)
    {
        /*
        communicator_ = std::make_unique<CollectiveCommunication<T>>( 
            input_shape, input_data, 
            output_shape, output_data, 
            distributor_->isGPUController(), *(distributor_->getWorkGroupCommunicator()) 
        );
        */

        communicator_ = std::make_unique<NonBlockingPtoPCommunication<T>>( 
            input_shape, input_data, 
            output_shape, output_data, 
            0, 
            *(distributor_->getWorkGroupCommunicator()) 
        );

        if (distributor_->isGPUController() )
        {
            best_batchsizes = findHybridBatchsize();
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

template<typename T>
std::pair<int64_t, int64_t> AIxeleratorService<T>::findHybridBatchsize()
{
    int num_workers = distributor_->getWorkGroupSize();
    int total_input_count = communicator_->getTotalInputCount();
    int total_output_count = communicator_->getTotalOutputCount();

    T* dummy_input_data = new T[total_input_count];
    for (int k = 0; k < total_input_count; k++)
    {
        dummy_input_data[k] = 13.37;
    }

    T* dummy_output_data = new T[total_output_count];
    for (int k = 0; k < total_output_count; k++)
    {
        dummy_output_data[k] = -42.24;
    }

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    std::ofstream myfile;
    std::string my_file_name = "hybrid_inference_" + std::to_string(my_rank) + ".dat";
    myfile.open(my_file_name);
    myfile << "host_fraction, batch_dim, batch_host, time_host, batch_device, time_device, hybrid_error" << std::endl;


    int num_samples = 1; 
    std::vector<int64_t> batchsizes_host(num_samples, -1);
    std::vector<int64_t> batchsizes_device(num_samples, -1);
    std::vector<double> times_host(num_samples, 0.0);
    std::vector<double> times_device(num_samples, 0.0);
    std::vector<double> hybrid_errors(num_samples, 1000000.0);

    
    for (int i = 0; i < num_samples; i++)
    {
        double batch_fraction_host = i * 0.01;
        int64_t batch_dim = input_shape_[0];
        int64_t batch_host = batch_dim * batch_fraction_host;
        int64_t batch_device = batch_dim - batch_host;
        batchsizes_host[i] = batch_host;
        batchsizes_device[i] = batch_device;

        // determine input data shape for host and device
        input_shape_host_ = input_shape_;
        input_shape_host_[0] = batch_host;
        input_shape_device_ = input_shape_;
        input_shape_device_[0] = batch_device;

        output_shape_host_ = output_shape_;
        output_shape_host_[0] = batch_host;
        output_shape_device_ = output_shape_;
        output_shape_device_[0] = batch_device;

        // setup data pointers accordingly
        input_data_host_ = &dummy_input_data[0];
        int64_t num_input_elem_per_batch = std::accumulate(std::next(input_shape_.begin(), 1), input_shape_.end(), 1, std::multiplies<int>());
        input_data_device_ = &dummy_input_data[batch_host * num_input_elem_per_batch];

        output_data_host_ = &dummy_output_data[0];
        int64_t num_output_elem_per_batch = std::accumulate(std::next(output_shape_.begin(), 1), output_shape_.end(), 1, std::multiplies<int>());
        output_data_device_ = &dummy_output_data[batch_host * num_output_elem_per_batch];

        std::unique_ptr<InferenceStrategy<T>> inference_host = createInferenceStrategy();
        if (batch_host > 0)
        {
            inference_host->init(batch_host, -1, model_file_name_, input_shape_host_, input_data_host_, output_shape_host_, output_data_host_);
        }

        std::unique_ptr<InferenceStrategy<T>> inference_device = createInferenceStrategy();
        if (batch_device > 0)
        {
            inference_device->init(batch_device, distributor_->getDeviceID(), model_file_name_, input_shape_device_, input_data_device_, output_shape_device_, output_data_device_); 
        }

        double start_host, end_host;
        try
        {
            if (batch_host > 0)
            {
                start_host = MPI_Wtime();
                inference_host->inference();
                end_host = MPI_Wtime();
            }
            else
            {
                start_host = 0.0;
                end_host = 0.0;
            }
        }
        catch(const std::runtime_error err)
        {
            std::cerr << "Runtime error for i = " << i << " in findHybridBatchsize (host): " << err.what() << std::endl;
            end_host = start_host;
        }
        catch(...)
        {
            std::cerr << "unexpected error during findHybridbBatchsize (host) i = " << i << std::endl;
            end_host = start_host;
        }
        double time_host = end_host - start_host;
        times_host[i] = time_host;
        
        double start_device, end_device;
        try
        {
            if (batch_device > 0)
            {
                start_device = MPI_Wtime();
                inference_device->inference();
                end_device = MPI_Wtime();
            }
            else
            {
                start_device = 0.0;
                end_device = 0.0;
            }
            
        }
        catch(const std::runtime_error err)
        {
            std::cerr << "Runtime error for i = " << i << " in findHybridBatchsize (device): " << err.what() << std::endl;
            end_device = start_device;
        }
        catch(...)
        {
            std::cerr << "unexpected error during findHybridbBatchsize (device) i = " << i << std::endl;
            end_device = start_device;
        }
        double time_device = end_device - start_device;
        times_device[i] = time_device;

        hybrid_errors[i] = std::abs(time_device - (time_host * num_workers));

        myfile << i << "," << batch_dim << "," << batch_host << "," << time_host << "," << batch_device << "," << time_device << "," << hybrid_errors[i] << std::endl;

        if (batch_host > 0)
        {
            inference_host.reset();
        }

        if (batch_device > 0)
        {
            inference_device.reset();
        }
    }

    auto min_error = std::min_element(hybrid_errors.begin(), hybrid_errors.end());
    auto min_index = std::distance(hybrid_errors.begin(), min_error);

    std::pair<int64_t, int64_t> best_batchsizes(batchsizes_host[min_index], batchsizes_device[min_index]);

    std::cout << "AIxeleratorService found best hybrid batch sizes to be: batchsize_host = " << batchsizes_host[min_index] << ", batchsize_device = " << batchsizes_device[min_index] << std::endl;

    myfile.close();
    delete[] dummy_input_data;
    delete[] dummy_output_data;

    return best_batchsizes;
}

template<typename T>
void AIxeleratorService<T>::initInferenceStrategy(std::pair<int64_t, int64_t> best_batchsizes)
{
    // determine batch dimension for host & device
    int64_t batch_host = best_batchsizes.first;
    int64_t batch_device = best_batchsizes.second;

    // determine input data shape for host and device
    input_shape_host_ = input_shape_;
    input_shape_host_[0] = batch_host;
    input_shape_device_ = input_shape_;
    input_shape_device_[0] = batch_device;

    output_shape_host_ = output_shape_;
    output_shape_host_[0] = batch_host;
    output_shape_device_ = output_shape_;
    output_shape_device_[0] = batch_device;

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
        /*
        communicator_ = std::make_unique<CollectiveCommunication<T>>(
            input_shape_device_, input_data_device_, 
            output_shape_device_, output_data_device_, 
            distributor_->isGPUController(), 
            *(distributor_->getWorkGroupCommunicator())
        );
        */

        communicator_ = std::make_unique<NonBlockingPtoPCommunication<T>>(
            input_shape_device_, input_data_device_, 
            output_shape_device_, output_data_device_, 
            0, 
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
    std::cout << "AIxeleratorService: gathering input data" << std::endl;
    if( communicator_ )
    {
        communicator_->gatherInputData();
    }

    std::cout << "AIxeleratorService: inference on device" << std::endl;
    if ( distributor_->isGPUController() && input_shape_device_[0] > 0)
    {
        inferencing_device_->inference();
    }

    std::cout << "AIxeleratorService: inference on host" << std::endl;
    if( input_shape_host_[0] > 0 )
    {
        inferencing_host_->inference();
    }

    std::cout << "AIxeleratorService: scattering output data" << std::endl;
    if( communicator_ )
    {
        communicator_->scatterOutputData();
    }
}

template class AIxeleratorService<float>;
template class AIxeleratorService<double>;