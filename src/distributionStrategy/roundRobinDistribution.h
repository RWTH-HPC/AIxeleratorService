#ifndef AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_ROUNDROBINDISTRIBUTION_H_
#define AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_ROUNDROBINDISTRIBUTION_H_

#include "distributionStrategy/distributionStrategy.h"

#include <vector>

class RoundRobinDistribution : public DistributionStrategy
{
    public:
        RoundRobinDistribution();
        ~RoundRobinDistribution();

        void createWorkgroups() override;

    private:
        int my_rank_;        
};

#endif