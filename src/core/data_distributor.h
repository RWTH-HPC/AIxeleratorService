#pragma once

#include <mpi.h>
#include <vector>

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
    std::vector<int> inputCounts_ {};
    std::vector<int> inputDisplacements_ {};
    std::vector<int> outputCounts_ {};
    std::vector<int> outputDisplacements_ {};
    std::vector<Dtype> inputTensorData_ {};
    std::vector<Dtype> outputTensorData_ {};

public:
    void gather();
    void scatter();
    int device_count();
};

} // namespace AIxelerator