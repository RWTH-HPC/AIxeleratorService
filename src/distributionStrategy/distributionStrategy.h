#ifndef AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_H_
#define AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_H_


#include <mpi.h>
#include <iostream>
#include <vector>

class DistributionStrategy
{
    public:
        virtual ~DistributionStrategy() = default;

        virtual void createWorkgroups() = 0;

        bool isGPUController(){ return is_gpu_controller_; }
        int getDeviceID(){ return my_gpu_device_; }
        int getNumDevicesTotal(){ return num_devices_total_; }

        MPI_Comm* getWorkGroupCommunicator(){return &work_group_comm_;}
        int getWorkGroupSize(){return workgroup_size_;}

    protected:
        MPI_Comm work_group_comm_;
        int workgroup_size_;
        bool is_gpu_controller_;
        int my_gpu_device_;
        int num_devices_total_;
};

#endif