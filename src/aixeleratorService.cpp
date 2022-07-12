#include "aixeleratorService.h"
#include <dlfcn.h>
#include <iostream>


AIxeleratorService::AIxeleratorService(AIFramework framework, std::string modelFile)
{
    std::cout << "AIxeleratorService created!" << std::endl;
    framework_ = framework;
    workGroupComm_ = MPI_COMM_WORLD;
    initWorkgroup(workGroupComm_);

    if (isGPUMaster_)
    {
        inputCounts_ = new int[nRanksWorkGroup_];
        inputDisplacements_ = new int[nRanksWorkGroup_];
        outputCounts_ = new int[nRanksWorkGroup_];
        outputDisplacements_ = new int[nRanksWorkGroup_];
    }

    // TODO: only GPU master ranks need to init Inference classes
    switch(framework)
    {
        case AIX_TORCH:
            torchInf_ = std::make_unique<torchInference>(modelFile, myGPUDevice_);
            break;
        case AIX_TENSORFLOW:
            std::cout << "Tensorflow Init (NYI)" << std::endl;
            break;
        default:
            std::cout << "Unknown Framework Init" << std::endl;
            break;
    }
}

AIxeleratorService::~AIxeleratorService()
{
    delete[] inputCounts_;
    delete[] inputDisplacements_;
    delete[] outputCounts_;
    delete[] outputDisplacements_;
    delete[] inputTensorData_;
    delete[] outputTensorData_;
}

int AIxeleratorService::deviceCount()
{
    int count = -1;
    void* cudaRT = dlopen("libcudart.so", RTLD_LAZY);
    if(cudaRT == NULL)
    {
        count = 0;
    }
    else{
        void (*getDeviceCount)(int*) = (void(*)(int*)) dlsym(cudaRT, "cudaGetDeviceCount");
        getDeviceCount(&count);  
    }
    return count;
}

void AIxeleratorService::initWorkgroup(MPI_Comm& workGroupComm)
{
    int myRank, nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    // figure out our local node rank
    MPI_Comm nodeCommunicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeCommunicator);
    int nodeRank, nodeSize;
    MPI_Comm_rank(nodeCommunicator, &nodeRank);
    MPI_Comm_size(nodeCommunicator, &nodeSize);
    MPI_Comm_free(&nodeCommunicator);

    printf("Rank %d/%d on its local machine is %d/%d\n", myRank, nProcs, nodeRank, nodeSize);

    int num_devices = deviceCount();
    printf("Rank %d/%d on its local machine is %d/%d and has access to %d devices!\n", myRank, nProcs, nodeRank, nodeSize, num_devices);
    // are we responsible for a gpu on our node?
    int gpuMaster = nodeRank < num_devices;

    if (gpuMaster) {
        printf("Rank %d/%d on its local machine is %d/%d is the GPU Master!\n", myRank, nProcs, nodeRank, nodeSize);
        isGPUMaster_ = true;
        myGPUDevice_ = nodeRank; // cuda device numbering starts with 1
        // TODO: move this as it depends on specific framework
        //c10::cuda::set_device(myGPUDevice_);
    }

    // figure out the total number of gpus across all nodes
    int totalGpuCount = 0;
    MPI_Allreduce(&gpuMaster, &totalGpuCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("Rank %d/%d knows that there is a total of %d gpus across all systems. \n", myRank, nProcs, totalGpuCount);

    // combine masters (and workers) into separate communicators, to enumerate them
    MPI_Comm workTypeComm;
    MPI_Comm_split(MPI_COMM_WORLD, gpuMaster, myRank, &workTypeComm);
    int workTypeRank, workTypeSize;
    MPI_Comm_rank(workTypeComm, &workTypeRank);
    MPI_Comm_size(workTypeComm, &workTypeSize);
    MPI_Comm_free(&workTypeComm);

    // round robin assignment of data to a gpu, making sure the gpu master is rank 0
    int color = workTypeRank % totalGpuCount;
    int order = gpuMaster ? 0 : myRank + totalGpuCount;
    printf("Rank %d/%d will be in group %d order %d\n", myRank, nProcs, color, order);

    // initialize the work group communicator
    MPI_Comm_split(MPI_COMM_WORLD, color, order, &workGroupComm);
    int workGroupRank, workGroupSize;
    MPI_Comm_rank(workGroupComm, &workGroupRank);
    MPI_Comm_size(workGroupComm, &workGroupSize);
    printf("Rank %d/%d got id of %d/%d within the workGroup\n", myRank, nProcs, workGroupRank, workGroupSize);
    nRanksWorkGroup_ = workGroupSize;
}

void AIxeleratorService::registerTensorShape(std::vector<int> &inputShape, std::vector<int> &outputShape)
{
    int inputCount = 0;
    for (const auto& value: inputShape)
    {
        inputCount += value;
    }

    MPI_Gather(&inputCount, 1, MPI_INT, inputCounts_, 1, MPI_INT, 0, workGroupComm_);

    if (isGPUMaster_)
    {
        inputDisplacements_[0] = 0;
        for(int i = 1; i < nRanksWorkGroup_; i++)
        {
            inputDisplacements_[i] = inputDisplacements_[i-1] + inputCounts_[i-1];
        }
    }

    // sum up individual inputSizes to get total input size
    int inputSizeTotal = 0;
    for(int i = 1; i < nRanksWorkGroup_; i++)
    {
        inputSizeTotal += inputCounts_[i];
    }

    inputTensorData_ = new double[inputSizeTotal];


    // same for output

    int outputCount = 0;
    for (const auto& value: outputShape)
    {
        outputCount += value;
    }

    MPI_Gather(&outputCount, 1, MPI_INT, outputCounts_, 1, MPI_INT, 0, workGroupComm_);

    if (isGPUMaster_)
    {
        outputDisplacements_[0] = 0;
        for(int i = 1; i < nRanksWorkGroup_; i++)
        {
            outputDisplacements_[i] = outputDisplacements_[i-1] + outputCounts_[i-1];
        }
    }

    // sum up individual inputSizes to get total input size
    int outputSizeTotal = 0;
    for(int i = 1; i < nRanksWorkGroup_; i++)
    {
        outputSizeTotal += outputCounts_[i];
    }

    outputTensorData_ = new double[outputSizeTotal];


    switch(framework_)
    {
        case AIX_TORCH:
            torchInf_->allocateTensors(inputShape, outputShape);
            break;
        case AIX_TENSORFLOW:
            std::cout << "Tensorflow registerDataSize (NYI)" << std::endl;
            break;
        default:
            std::cout << "Unknown Framework registerDataSize" << std::endl;
            break;
    }
        
}

void AIxeleratorService::gatherTensorData(double* input, int inputCount)
{
    MPI_Gatherv(input, inputCount, MPI_DOUBLE, inputTensorData_, inputCounts_, inputDisplacements_, MPI_DOUBLE, 0, workGroupComm_);
}

void AIxeleratorService::scatterTensorData(double* output, int outputCount)
{
    MPI_Scatterv(output, outputCounts_, outputDisplacements_, MPI_DOUBLE, outputTensorData_, outputCount, MPI_DOUBLE, 0, workGroupComm_);
}

void AIxeleratorService::inference(double* input, int inputCount, double* output, int outputCount)
{

    gatherTensorData(input, inputCount);
    int batchSize = 1000; // TODO: Properly calculate this value

    if (isGPUMaster_)
    {
        switch(framework_)
        {
            case AIX_TORCH:
                torchInf_->forward(inputTensorData_, outputTensorData_, batchSize);
                break;
            case AIX_TENSORFLOW:
                std::cout << "Tensorflow Inference (NYI)" << std::endl;
                break;
            default:
                std::cout << "Unknown Framework Inference" << std::endl;
                break;
        }
    }

    scatterTensorData(output, outputCount);
}

