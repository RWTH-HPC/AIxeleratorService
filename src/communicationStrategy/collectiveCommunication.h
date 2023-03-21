#ifndef AIXELERATORSERVICE_COMMUNICATIONSTRATEGY_COLLECTIVECOMUNICATION_H_
#define AIXELERATORSERVICE_COMMUNICATIONSTRATEGY_COLLECTIVECOMUNICATION_H_

#include "communicationStrategy/communicationStrategy.h"

#include <mpi.h>
#include <vector>

class CollectiveCommunication : public CommunicationStrategy
{
    public:
        CollectiveCommunication() = delete;
        CollectiveCommunication(
            std::vector<int64_t> input_shape, double* input_data, 
            std::vector<int64_t> output_shape, double* output_data, 
            bool is_device_controller, MPI_Comm work_group_comm
        );
        ~CollectiveCommunication();

        void gatherInputData() override;
        void scatterOutputData() override;

        void setInputData(int input_sendcount, double* input_data);
        void setOutputData(int output_sendcount, double* output_data);

        int getTotalInputCount(){return total_input_count_;}
        int getTotalOutputCount(){return total_output_count_;}

    private:
        bool is_device_controller_;

        int workgroup_size_;

        MPI_Comm work_group_comm_;
        double* input_data_worker_;

        int input_sendcount_;
        int total_input_count_;
        std::vector<int> input_recvcounts_;
        std::vector<int> input_displs_;

        double* output_data_worker_;       

        int output_sendcount_;
        int total_output_count_;
        std::vector<int> output_recvcounts_;
        std::vector<int> output_displs_;
};

#endif