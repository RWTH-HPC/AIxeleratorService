#ifndef AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_H_
#define AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_H_

class DistributionStrategy
{
    public:
        virtual ~DistributionStrategy() = default;

        virtual void createWorkgroups() = 0;
        virtual void gatherInputData() = 0;
        virtual void scatterOutputData() = 0;
};

#endif