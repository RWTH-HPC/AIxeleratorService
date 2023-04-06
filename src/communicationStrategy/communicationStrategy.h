#ifndef AIXELERATORSERVICE_COMMUNICATIONSTRATEGY_H_
#define AIXELERATORSERVICE_COMMUNICATIONSTRATEGY_H_

#include <vector>
#include <cstdint>

class CommunicationStrategy
{
    public:
        virtual ~CommunicationStrategy() = default;

        virtual void gatherInputData() = 0;
        virtual void scatterOutputData() = 0;

        double* getInputDataController(){ return input_data_controller_; }
        double* getOutputDataController(){ return output_data_controller_; }
    
        std::vector<int64_t> getInputShapeController(){return input_shape_controller_;}
        std::vector<int64_t> getOutputShapeController(){return output_shape_controller_;}

        int getTotalInputCount(){return total_input_count_;}
        int getTotalOutputCount(){return total_output_count_;}

    protected:
        double* input_data_controller_;
        double* output_data_controller_;

        std::vector<int64_t> input_shape_controller_;
        std::vector<int64_t> output_shape_controller_; 

        int total_input_count_;
        int total_output_count_;       
};

#endif