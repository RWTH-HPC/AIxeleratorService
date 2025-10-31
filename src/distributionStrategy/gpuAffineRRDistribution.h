#ifndef AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_GPUAFFINERRDISTRIBUTION_H_
#define AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_GPUAFFINERRDISTRIBUTION_H_

#include "distributionStrategy/distributionStrategy.h"

#include <vector>

#include <hwloc.h>
#include <vector>
#include <map>
#include <iostream>
#include <limits.h>

struct GPUInfo {
    int index;          // GPU logical index (0,1,...)
    int numa_node;      // NUMA node this GPU is attached to
};

class GPUAffineRRDistribution : public DistributionStrategy
{
    public:
        GPUAffineRRDistribution(MPI_Comm app_comm);
        ~GPUAffineRRDistribution();

        void createWorkgroups() override;
        int getRankNUMANode();
        std::vector<GPUInfo> discoverGPUs();

    private:
        int my_rank_; 
        MPI_Comm app_comm_;       
};

#endif