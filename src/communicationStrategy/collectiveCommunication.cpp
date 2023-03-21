#include "communicationStrategy/collectiveCommunication.h"

#include <iostream>
#include <numeric>

CollectiveCommunication::CollectiveCommunication(
    std::vector<int64_t> input_shape, double* input_data, 
    std::vector<int64_t> output_shape, double* output_data, 
    bool is_device_controller, MPI_Comm work_group_comm)
{

    is_device_controller_ = is_device_controller;
    work_group_comm_ = work_group_comm;
    MPI_Comm_size(work_group_comm_, &workgroup_size_);

    int input_sendcount = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
    int output_sendcount = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

    setInputData(input_sendcount, input_data);
    setOutputData(output_sendcount, output_data);

    // determine number of samples over all processes
    int batch_dim = input_shape[0];
    int total_batch_dim = 0;
    MPI_Reduce(&batch_dim, &total_batch_dim, 1, MPI_INT, MPI_SUM, 0, work_group_comm_);
    input_shape_controller_ = input_shape;
    input_shape_controller_[0] = total_batch_dim;
    output_shape_controller_ = output_shape;
    output_shape_controller_[0] = total_batch_dim;

    if( is_device_controller_ )
    {
        input_recvcounts_.resize(workgroup_size_, -1);
        input_displs_.resize(workgroup_size_, -1);

        output_recvcounts_.resize(workgroup_size_, -1);
        output_displs_.resize(workgroup_size_, -1);
    }
    else
    {
        input_recvcounts_.resize(0);
        input_displs_.resize(0);

        output_recvcounts_.resize(0);
        output_displs_.resize(0);
    }
    
    MPI_Gather(&input_sendcount_, 1, MPI_INT, input_recvcounts_.data(), 1, MPI_INT, 0, work_group_comm_);
    MPI_Gather(&output_sendcount_, 1, MPI_INT, output_recvcounts_.data(), 1, MPI_INT, 0, work_group_comm_);
    
    if( is_device_controller_ )
    {
        input_displs_[0] = 0;
        output_displs_[0] = 0;
        for(int i = 1; i < workgroup_size_; i++)
        {
            input_displs_[i] = input_displs_[i-1] + input_recvcounts_[i-1];
            output_displs_[i] = output_displs_[i-1] + output_recvcounts_[i-1];
        }  

        total_input_count_ = std::accumulate(input_recvcounts_.begin(), input_recvcounts_.end(), 0);
        total_output_count_ = std::accumulate(output_recvcounts_.begin(), output_recvcounts_.end(), 0);

        input_data_controller_ = new double[total_input_count_];
        output_data_controller_ = new double[total_output_count_];
    }
}

CollectiveCommunication::~CollectiveCommunication()
{
    if( is_device_controller_ )
    {
        delete[] input_data_controller_;
        delete[] output_data_controller_;
    }
}

void CollectiveCommunication::setInputData(int input_sendcount, double* input_data)
{
    input_sendcount_ = input_sendcount;
    input_data_worker_ = input_data;
}

void CollectiveCommunication::setOutputData(int output_sendcount, double* output_data)
{
    output_sendcount_ = output_sendcount;
    output_data_worker_ = output_data;
}


void CollectiveCommunication::gatherInputData()
{
    MPI_Gatherv(input_data_worker_, input_sendcount_, MPI_DOUBLE, input_data_controller_, input_recvcounts_.data(), input_displs_.data(), MPI_DOUBLE, 0, work_group_comm_);  
}

void CollectiveCommunication::scatterOutputData()
{
    MPI_Scatterv(output_data_controller_, output_recvcounts_.data(), output_displs_.data(), MPI_DOUBLE, output_data_worker_, output_sendcount_, MPI_DOUBLE, 0, work_group_comm_);
}
