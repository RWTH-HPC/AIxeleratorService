#ifndef AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_ROUNDROBINDISTRIBUTION_H_
#define AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_ROUNDROBINDISTRIBUTION_H_

#include "distributionStrategy/distributionStrategy.h"

#include <mpi.h>
#include <vector>

class RoundRobinDistribution : public DistributionStrategy
{
    public:
        RoundRobinDistribution() = delete;
        RoundRobinDistribution(std::vector<int64_t> input_shape, double* input_data, std::vector<int64_t> output_shape, double* output_data);
        ~RoundRobinDistribution();

        void createWorkgroups() override;
        void gatherInputData() override;
        void scatterOutputData() override;

        void setInputData(int input_sendcount, double* input_data);
        void setOutputData(int output_sendcount, double* output_data);

        void setInputSizes(int sendcount, std::vector<int> recvcounts, std::vector<int> displs);
        void setOutputSizes(int sendcount, std::vector<int> recvcounts, std::vector<int> displs);

        int getTotalInputCount(){return total_input_count_;}
        int getTotalOutputCount(){return total_output_count_;}

    private:
        MPI_Comm work_group_comm_;
        int workgroup_size_;

        double* input_data_worker_;
        
        int input_sendcount_;
        int total_input_count_;
        std::vector<int> input_recvcounts_;
        std::vector<int> input_displs_;

        double* output_data_worker_;
        int my_rank_;        

        int output_sendcount_;
        int total_output_count_;
        std::vector<int> output_recvcounts_;
        std::vector<int> output_displs_;

};

#endif