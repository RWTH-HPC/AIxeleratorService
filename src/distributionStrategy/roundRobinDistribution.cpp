#include "distributionStrategy/roundRobinDistribution.h"

#include "utils/deviceCount.h"

#include <iostream>
#include <numeric>

RoundRobinDistribution::RoundRobinDistribution(int input_sendcount, double* input_data, int output_sendcount, double* output_data)
{
    work_group_comm_ = MPI_COMM_WORLD;
    createWorkgroups();

    input_sendcount_ = input_sendcount;
    input_data_worker_ = input_data;

    output_sendcount_ = output_sendcount;
    output_data_worker_ = output_data;

    if(isGPUController())
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
    
    MPI_Gather(&input_sendcount_, 1, MPI_INT, input_recvcounts_.data(), workgroup_size_, MPI_INT, 0, work_group_comm_);
    MPI_Gather(&output_sendcount_, 1, MPI_INT, output_recvcounts_.data(), workgroup_size_, MPI_INT, 0, work_group_comm_);
    
    if(isGPUController())
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


    //setInputSizes(input_sendcount, input_recvcounts, input_displs);
    //setOutputSizes(output_sendcount, output_recvcounts, output_displs);
}

void RoundRobinDistribution::createWorkgroups()
{
    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // figure out our local node rank
    MPI_Comm node_communicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_communicator);
    int node_rank, node_size;
    MPI_Comm_rank(node_communicator, &node_rank);
    MPI_Comm_size(node_communicator, &node_size);
    MPI_Comm_free(&node_communicator);

    std::cout << "Rank " << my_rank << "/" << num_procs << "on its local machine is " << node_rank << "/" << node_size << std::endl; 

    int num_devices = aixelerator_service::utils::deviceCount();

    std::cout << "Rank " << my_rank << "/" << num_procs << "on its local machine is " << node_rank << "/" << node_size << "and has access to " << num_devices << " devices!" << std::endl; 

    bool is_gpu_controller_ = node_rank < num_devices;

    if (is_gpu_controller_)
    {
         std::cout << "Rank " << my_rank << "/" << num_procs << "on its local machine is " << node_rank << "/" << node_size << "is the GPU controller!" << std::endl; 
         my_gpu_device_ = node_rank;
    }

    // figure out the total number of gpus across all nodes
    int total_gpu_count = 0;
    int my_num_devices = is_gpu_controller_ ? 1 : 0;
    MPI_Allreduce(&my_num_devices, &total_gpu_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout << "Rank " << my_rank << "/" << num_procs << "knows that there is a total of " << total_gpu_count << " GPUs across all systems" << std::endl;

    // combine controllers (and workers) into separate communicators, to enumerate them
    MPI_Comm work_type_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_num_devices, my_rank, &work_type_comm);
    int work_type_rank, work_type_size;
    MPI_Comm_rank(work_type_comm, &work_type_rank);
    MPI_Comm_size(work_type_comm, &work_type_size);
    MPI_Comm_free(&work_type_comm);

    // round robin assignment of data to a gpu, making sure the gpu master is rank 0
    int color = work_type_rank % total_gpu_count;
    int order = is_gpu_controller_ ? 0 : my_rank + total_gpu_count;
    std::cout << "Rank " << my_rank << "/" << num_procs << "will be in group " << color << " order " << order << std::endl;

    // initialize the work group communicator
    MPI_Comm_split(MPI_COMM_WORLD, color, order, &work_group_comm_);
    int work_group_rank, work_group_size;
    MPI_Comm_rank(work_group_comm_, &work_group_rank);
    MPI_Comm_size(work_group_comm_, &work_group_size);
    workgroup_size_ = work_group_size;
    std::cout << "Rank " << my_rank << "/" << num_procs << "got id of " << work_group_rank << "/" << work_group_size << std::endl;
}

void RoundRobinDistribution::setInputSizes(int sendcount, std::vector<int> recvcounts, std::vector<int> displs)
{
    input_sendcount_ = sendcount;
    input_recvcounts_ = recvcounts;
    input_displs_ = displs;
}

void RoundRobinDistribution::setOutputSizes(int sendcount, std::vector<int> recvcounts, std::vector<int> displs)
{
    output_sendcount_ = sendcount;
    output_recvcounts_ = recvcounts;
    output_displs_ = displs;
}

void RoundRobinDistribution::gatherInputData()
{
    MPI_Gatherv(input_data_worker_, input_sendcount_, MPI_DOUBLE, input_data_controller_, input_recvcounts_.data(), input_displs_.data(), MPI_DOUBLE, 0, work_group_comm_);
}

void RoundRobinDistribution::scatterOutputData()
{
    MPI_Scatterv(output_data_controller_, output_recvcounts_.data(), output_displs_.data(), MPI_DOUBLE, output_data_worker_, output_sendcount_, MPI_DOUBLE, 0, work_group_comm_);
}