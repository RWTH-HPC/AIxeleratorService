#include "distributionStrategy/roundRobinDistribution.h"

#include "utils/deviceCount.h"

#include <iostream>
#include <numeric>


RoundRobinDistribution::RoundRobinDistribution() 
{
    work_group_comm_ = MPI_COMM_WORLD;
    createWorkgroups();
}

RoundRobinDistribution::~RoundRobinDistribution()
{

}

void RoundRobinDistribution::createWorkgroups()
{
    int my_rank, num_procs;
    int err;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    my_rank_ = my_rank;

    // figure out our local node rank
    MPI_Comm node_communicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_communicator);
    int node_rank, node_size;
    MPI_Comm_rank(node_communicator, &node_rank);
    MPI_Comm_size(node_communicator, &node_size);
    MPI_Comm_free(&node_communicator);

    std::cout << "Rank " << my_rank << "/" << num_procs << " on its local machine is " << node_rank << "/" << node_size << std::endl; 

    int num_devices = aixelerator_service::utils::deviceCount();

    std::cout << "Rank " << my_rank << "/" << num_procs << " on its local machine is " << node_rank << "/" << node_size << " and has access to " << num_devices << " devices!" << std::endl; 

    is_gpu_controller_ = node_rank < num_devices;

    if (is_gpu_controller_)
    {
         std::cout << "Rank " << my_rank << "/" << num_procs << " on its local machine is " << node_rank << "/" << node_size << " is the GPU controller!" << std::endl; 
         my_gpu_device_ = node_rank;
    }
    else
    {
        my_gpu_device_ = -1;
    }

    // figure out the total number of gpus across all nodes
    num_devices_total_ = 0;
    int my_num_devices = is_gpu_controller_ ? 1 : 0;
    MPI_Allreduce(&my_num_devices, &num_devices_total_, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout << "Rank " << my_rank << "/" << num_procs << " knows that there is a total of " << num_devices_total_ << " GPUs across all systems" << std::endl;

    if ( num_devices_total_ > 0)
    {
        // combine controllers (and workers) into separate communicators, to enumerate them
        MPI_Comm work_type_comm;
        MPI_Comm_split(MPI_COMM_WORLD, my_num_devices, my_rank, &work_type_comm);
        int work_type_rank, work_type_size;
        MPI_Comm_rank(work_type_comm, &work_type_rank);
        MPI_Comm_size(work_type_comm, &work_type_size);
        MPI_Comm_free(&work_type_comm);

        // round robin assignment of data to a gpu, making sure the gpu master is rank 0
        int color = work_type_rank % num_devices_total_;
        int order = is_gpu_controller_ ? 0 : my_rank + num_devices_total_;
        std::cout << "Rank " << my_rank << "/" << num_procs << " will be in group " << color << " order " << order << std::endl;

        // initialize the work group communicator
        err = MPI_Comm_split(MPI_COMM_WORLD, color, order, &work_group_comm_);
        if ( err != MPI_SUCCESS )
        {
            std::cout << "Rank " << my_rank << ": Error when splitting workgroup communicator " << work_group_comm_  << std::endl;
        }
        int work_group_rank, work_group_size;
        MPI_Comm_rank(work_group_comm_, &work_group_rank);
        MPI_Comm_size(work_group_comm_, &work_group_size);
        workgroup_size_ = work_group_size;
        std::cout << "Rank " << my_rank << "/" << num_procs << " got id of " << work_group_rank << "/" << work_group_size << std::endl;
    }
    else
    {
        workgroup_size_ = 0;
        work_group_comm_ = MPI_COMM_WORLD;
    } 
}
