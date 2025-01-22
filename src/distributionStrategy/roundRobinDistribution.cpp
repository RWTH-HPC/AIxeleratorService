#include "distributionStrategy/roundRobinDistribution.h"

#include "utils/deviceCount.h"

#include <iostream>
#include <numeric>


RoundRobinDistribution::RoundRobinDistribution(MPI_Comm app_comm) 
{
    app_comm_ = app_comm;
    work_group_comm_ = app_comm_;
    /*
     * if AIxeleratorService is used together with PhyDLL or MLLIB in an MPMD 
     * run (i.e. app_comm_ will be different from MPI_COMM_WORLD), then we do
     * not want to use the GPU but leave it for PhyDLL or MLLIB.
     */
    if (app_comm_ == MPI_COMM_WORLD) {
        createWorkgroups();
    }
    else{
        workgroup_size_ = 0;
        num_devices_total_ = 0;
        is_gpu_controller_ = false;
    }
}

RoundRobinDistribution::~RoundRobinDistribution()
{

}

void RoundRobinDistribution::createWorkgroups()
{
    int my_rank, num_procs;
    int err;
    MPI_Comm_rank(app_comm_, &my_rank);
    MPI_Comm_size(app_comm_, &num_procs);
    my_rank_ = my_rank;

    // figure out our local node rank
    MPI_Comm node_communicator;
    MPI_Comm_split_type(app_comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_communicator);
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
    MPI_Allreduce(&my_num_devices, &num_devices_total_, 1, MPI_INT, MPI_SUM, app_comm_);
    std::cout << "Rank " << my_rank << "/" << num_procs << " knows that there is a total of " << num_devices_total_ << " GPUs across all systems" << std::endl;

    if ( num_devices_total_ > 0)
    {
        // combine controllers (and workers) into separate communicators, to enumerate them
        MPI_Comm work_type_comm;
        MPI_Comm_split(app_comm_, my_num_devices, my_rank, &work_type_comm);
        int work_type_rank, work_type_size;
        MPI_Comm_rank(work_type_comm, &work_type_rank);
        MPI_Comm_size(work_type_comm, &work_type_size);
        MPI_Comm_free(&work_type_comm);

        // round robin assignment of data to a gpu, making sure the gpu master is rank 0
        int color = work_type_rank % num_devices_total_;
        int order = is_gpu_controller_ ? 0 : my_rank + num_devices_total_;
        std::cout << "Rank " << my_rank << "/" << num_procs << " will be in group " << color << " order " << order << std::endl;

        // initialize the work group communicator
        err = MPI_Comm_split(app_comm_, color, order, &work_group_comm_);
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
        work_group_comm_ = app_comm_;
    } 
}
