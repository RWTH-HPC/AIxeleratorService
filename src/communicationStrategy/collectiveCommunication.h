#ifndef AIXELERATORSERVICE_COMMUNICATIONSTRATEGY_COLLECTIVECOMUNICATION_H_
#define AIXELERATORSERVICE_COMMUNICATIONSTRATEGY_COLLECTIVECOMUNICATION_H_

#include "communicationStrategy/communicationStrategy.h"

#include <mpi.h>
#include <vector>

template<typename T>
class CollectiveCommunication : public CommunicationStrategy<T>
{
    public:
        CollectiveCommunication() = delete;
        CollectiveCommunication(
            std::vector<int64_t> input_shape, T* input_data, 
            std::vector<int64_t> output_shape, T* output_data, 
            bool is_device_controller, MPI_Comm work_group_comm
        );
        ~CollectiveCommunication();

        void gatherInputData() override;
        void scatterOutputData() override;

        void setInputData(int input_sendcount, T* input_data);
        void setOutputData(int output_sendcount, T* output_data);

    private:
        bool is_device_controller_;

        int workgroup_size_;

        MPI_Comm work_group_comm_;
        MPI_Datatype dtype_;
        T* input_data_worker_;

        int input_sendcount_;
        std::vector<int> input_recvcounts_;
        std::vector<int> input_displs_;

        T* output_data_worker_;       

        int output_sendcount_;
        std::vector<int> output_recvcounts_;
        std::vector<int> output_displs_;
};

#endif