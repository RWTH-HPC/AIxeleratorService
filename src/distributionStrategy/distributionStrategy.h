#ifndef AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_H_
#define AIXELERATORSERVICE_DISTRIBUTIONSTRATEGY_H_

#include <iostream>
#include <vector>

class DistributionStrategy
{
    public:
        virtual ~DistributionStrategy() = default;

        virtual void createWorkgroups() = 0;
        virtual void gatherInputData() = 0;
        virtual void scatterOutputData() = 0;

        bool isGPUController(){ return is_gpu_controller_; }
        int getDeviceID(){ return my_gpu_device_; }

        double* getInputDataController(){ return input_data_controller_; }
        double* getOutputDataController(){ return output_data_controller_; }
    
        std::vector<int64_t> getInputShapeController(){return input_shape_controller_;}
        std::vector<int64_t> getOutputShapeController(){return output_shape_controller_;}

    protected:
        bool is_gpu_controller_;
        int my_gpu_device_;

        double* input_data_controller_;
        double* output_data_controller_;

        std::vector<int64_t> input_shape_controller_;
        std::vector<int64_t> output_shape_controller_;
};

#endif