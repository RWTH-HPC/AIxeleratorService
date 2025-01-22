#include "communicationStrategy/nonBlockingPtoPCommunication.h"

#include <numeric>
#include <iostream>

template<typename T>
MPI_Datatype getTypeFromTemplate()
{
    if (std::is_same<T, float>::value)
    {
        return MPI_FLOAT;
    }

    if (std::is_same<T, double>::value)
    {
        return MPI_DOUBLE;
    }

    std::cerr << "ERROR: Could not get MPI_Datatype from template - defaulting to MPI_FLOAT!" << std::endl;
    // TODO(fabian): find a nicer way to handle error. Maybe use std::optional as return type?
    return MPI_FLOAT;
}


template<typename T>
NonBlockingPtoPCommunication<T>::NonBlockingPtoPCommunication(
    std::vector<int64_t> input_shape, T* input_data, 
    std::vector<int64_t> output_shape, T* output_data, 
    int device_controller_rank, MPI_Comm work_group_comm
){
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);
    device_controller_rank_ = device_controller_rank;
    work_group_comm_ = work_group_comm;
    MPI_Comm_rank(work_group_comm_, &my_workgroup_rank_);
    is_device_controller_ = (my_workgroup_rank_ == device_controller_rank_);
    MPI_Comm_size(work_group_comm_, &workgroup_size_);
    dtype_ = getTypeFromTemplate<T>();

    int input_sendcount = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
    int output_sendcount = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

    setInputData(input_sendcount, input_data);
    setOutputData(output_sendcount, output_data);

    // determine number of samples over all processes
    int batch_dim = input_shape[0];
    int total_batch_dim = 0;
    MPI_Reduce(&batch_dim, &total_batch_dim, 1, MPI_INT, MPI_SUM, 0, work_group_comm_);
    this->total_input_samples_ = total_batch_dim;

    this->input_shape_controller_ = input_shape;
    this->input_shape_controller_[0] = total_batch_dim;
    this->output_shape_controller_ = output_shape;
    this->output_shape_controller_[0] = total_batch_dim;

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

        this->total_input_count_ = std::accumulate(input_recvcounts_.begin(), input_recvcounts_.end(), 0);
        this->total_output_count_ = std::accumulate(output_recvcounts_.begin(), output_recvcounts_.end(), 0);

        this->input_data_controller_ = new T[this->total_input_count_];
        this->output_data_controller_ = new T[this->total_output_count_];

        input_requests_.resize(workgroup_size_);
        output_requests_.resize(workgroup_size_);
        input_statuses_.resize(workgroup_size_);
        output_statuses_.resize(workgroup_size_);
    }
}

template<typename T>
NonBlockingPtoPCommunication<T>::~NonBlockingPtoPCommunication()
{
    if( is_device_controller_ )
    {
        delete[] this->input_data_controller_;
        delete[] this->output_data_controller_;
    }
}

template<typename T>
void NonBlockingPtoPCommunication<T>::setInputData(int input_sendcount, T* input_data)
{
    input_sendcount_ = input_sendcount;
    input_data_worker_ = input_data;
}

template<typename T>
void NonBlockingPtoPCommunication<T>::setOutputData(int output_sendcount, T* output_data)
{
    output_sendcount_ = output_sendcount;
    output_data_worker_ = output_data;
}

template<typename T>
void NonBlockingPtoPCommunication<T>::gatherInputData()
{
    if( is_device_controller_ )
    {     
        for(int i = 0; i < workgroup_size_; i++)
        {
            int offset = input_displs_[i];
            int recv_count = input_recvcounts_[i];
            MPI_Irecv(&(this->input_data_controller_[offset]), recv_count, dtype_, i, i, work_group_comm_, &input_requests_[i]);
        }
    }

    MPI_Isend(input_data_worker_, input_sendcount_, dtype_, device_controller_rank_, my_workgroup_rank_, work_group_comm_, &send_req_);

    if( is_device_controller_ )
    {
        MPI_Waitall(workgroup_size_, input_requests_.data(), input_statuses_.data());
    }
}

template<typename T>
void NonBlockingPtoPCommunication<T>::scatterOutputData()
{
    MPI_Irecv(output_data_worker_, output_sendcount_, dtype_, device_controller_rank_, my_workgroup_rank_, work_group_comm_, &recv_req_);

    if ( is_device_controller_ )
    {
        for(int i = 0; i < workgroup_size_; i++)
        {
            int offset = output_displs_[i];
            int send_count = output_recvcounts_[i];
            MPI_Isend(&(this->output_data_controller_[offset]), send_count, dtype_, i, i, work_group_comm_, &output_requests_[i]);
        }  
    }

    MPI_Wait(&recv_req_, &recv_status_);
}

template class NonBlockingPtoPCommunication<float>;
template class NonBlockingPtoPCommunication<double>;