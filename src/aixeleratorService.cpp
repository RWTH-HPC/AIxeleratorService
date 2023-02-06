#include "aixeleratorService.h"

#include "inferenceStrategy/torchInference.h"
#include "inferenceStrategy/tensorflowInference.h"
#include "distributionStrategy/roundRobinDistribution.h"

#include <numeric>
#include <iostream>

AIxeleratorService::AIxeleratorService()
{
    batchsize_ = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);
}

void AIxeleratorService::registerModel(std::string model_file)
{
    // TODO: only GPU-controller ranks should create unique pointers with inference strategy. This requires to initalize distribution in constructor
    model_file_name_ = model_file;

    if ( model_file_name_.ends_with(std::string_view(".pt")) )
    {
        inferencing_ = std::make_unique<TorchInference>();
        framework_ = AIX_TORCH;
    }
    else if ( model_file_name_.ends_with(std::string_view(".pb")) || model_file_name_.ends_with(std::string_view(".tf")) )
    {
        inferencing_ = std::make_unique<TensorflowInference>();
        framework_ = AIX_TENSORFLOW;
    }
    else
    {
        std::cerr << "Error: AIxeleratorService does not support format of model file: " << model_file_name_ << std::endl;
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