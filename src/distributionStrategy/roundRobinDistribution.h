#ifndef AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_ROUNDROBINDISTRIBUTION_H_
#define AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_ROUNDROBINDISTRIBUTION_H_

#include "distributionStrategy/distributionStrategy.h"

#include <vector>


class RoundRobinDistribution : public DistributionStrategy
{
    public:
        RoundRobinDistribution(MPI_Comm app_comm);
        ~RoundRobinDistribution();

        void createWorkgroups() override;

    private:
        int my_rank_; 
        MPI_Comm app_comm_;       
};

#endif