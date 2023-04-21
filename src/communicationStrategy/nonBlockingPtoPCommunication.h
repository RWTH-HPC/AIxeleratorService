#ifndef AIXELERATORSERVICE_COMMUNICATIONSTRATEGY_NONBLOCKINGPTOPCOMMUNICATION_H_
#define AIXELERATORSERVICE_COMMUNICATIONSTRATEGY_NONBLOCKINGPTOPCOMMUNICATION_H_

#include "communicationStrategy/communicationStrategy.h"

#include <mpi.h>
#include <vector>

template<typename T>
class NonBlockingPtoPCommunication : public CommunicationStrategy<T>
{
    public:
        NonBlockingPtoPCommunication() = delete;
        NonBlockingPtoPCommunication(
            std::vector<int64_t> input_shape, T* input_data, 
            std::vector<int64_t> output_shape, T* output_data, 
            int device_controller_rank, MPI_Comm work_group_comm
        );
        ~NonBlockingPtoPCommunication();

        void gatherInputData() override;
        void scatterOutputData() override;

        void setInputData(int input_sendcount, T* input_data);
        void setOutputData(int output_sendcount, T* output_data);

    private:
        bool is_device_controller_;
        int my_rank_;
        int my_workgroup_rank_;
        int workgroup_size_;
        int device_controller_rank_;
        MPI_Request send_req_;
        MPI_Request recv_req_;
        MPI_Status recv_status_;
        std::vector<MPI_Request> input_requests_;
        std::vector<MPI_Request> output_requests_;
        std::vector<MPI_Status> input_statuses_;
        std::vector<MPI_Status> output_statuses_;
        MPI_Datatype dtype_;
        MPI_Comm work_group_comm_;

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