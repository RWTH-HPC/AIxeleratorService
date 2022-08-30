#pragma once

#include <memory>
#include <mpi.h>
#include <vector>

#include "torch/script.h"

namespace AIxelerator {

class RoundRobinStrategy;

template <typename Dtype = double, typename strategy = RoundRobinStrategy>
class DataDistributor {
private:
    MPI_Comm workGroupComm_;
    bool isGPUMaster_;
    int myGPUDevice_;
    int nRanksWorkGroup_;

    // only allocated on GPUMaster ranks
    int inputSizeTotal_ {};
    int outputSizeTotal_ {};
    int device_count_ {};
    std::vector<int> inputCounts_ {};
    std::vector<int> inputDisplacements_ {};
    std::vector<int> outputCounts_ {};
    std::vector<int> outputDisplacements_ {};
    std::vector<Dtype> inputTensorData_ {};
    std::vector<Dtype> outputTensorData_ {};
    std::shared_ptr<torch::Tensor> inputTensor_ = nullptr;
    std::shared_ptr<torch::Tensor> outputTensor_ = nullptr;

public:
    DataDistributor();
    void gather();
    void scatter();
    int device_count();

private:
    void to_input();
    void from_output();
};

} // namespace AIxelerator