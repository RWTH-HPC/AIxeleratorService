#ifndef AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_ROUNDROBINDISTRIBUTION_H_
#define AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_ROUNDROBINDISTRIBUTION_H_

#include "distributionStrategy/distributionStrategy.h"

#include <mpi.h>
#include <vector>

class RoundRobinDistribution : public DistributionStrategy
{
    public:
        RoundRobinDistribution() = delete;
        RoundRobinDistribution(int input_sendcount, double* input_data, int output_sendcount, double* output_data);
        ~RoundRobinDistribution() = default;

        void createWorkgroups() override;
        void gatherInputData() override;
        void scatterOutputData() override;

        bool isGPUController(){ return is_gpu_controller_; }

        void setInputSizes(int sendcount, std::vector<int> recvcounts, std::vector<int> displs);
        void setOutputSizes(int sendcount, std::vector<int> recvcounts, std::vector<int> displs);

        double* getInputDataController(){return input_data_controller_;}
        double* getOutputDataController(){return output_data_controller_;}

        int getTotalInputCount(){return total_input_count_;}
        int getTotalOutputCount(){return total_output_count_;}

    private:
        bool is_gpu_controller_;
        int my_gpu_device_;
        MPI_Comm work_group_comm_;
        int workgroup_size_;

        double* input_data_worker_;
        double* input_data_controller_;

        int input_sendcount_;
        int total_input_count_;
        std::vector<int> input_recvcounts_;
        std::vector<int> input_displs_;

        double* output_data_worker_;
        double* output_data_controller_;

        int output_sendcount_;
        int total_output_count_;
        std::vector<int> output_recvcounts_;
        std::vector<int> output_displs_;

};

#endif