#ifndef AIXELERATORSERVICE_COMMUNICATIONSTRATEGY_H_
#define AIXELERATORSERVICE_COMMUNICATIONSTRATEGY_H_

#include <vector>
#include <cstdint>

template<typename T>
class CommunicationStrategy
{
    public:
        virtual ~CommunicationStrategy() = default;

        virtual void gatherInputData() = 0;
        virtual void scatterOutputData() = 0;

        T* getInputDataController(){ return input_data_controller_; }
        T* getOutputDataController(){ return output_data_controller_; }
    
        std::vector<int64_t> getInputShapeController(){return input_shape_controller_;}
        std::vector<int64_t> getOutputShapeController(){return output_shape_controller_;}

        int getTotalInputCount(){return total_input_count_;}
        int getTotalOutputCount(){return total_output_count_;}

        int getTotalInputSamples(){return total_input_samples_;}

    protected:
        T* input_data_controller_;
        T* output_data_controller_;

        std::vector<int64_t> input_shape_controller_;
        std::vector<int64_t> output_shape_controller_; 

        int64_t total_input_count_;
        int64_t total_output_count_;  

        int64_t total_input_samples_;     
};

#endif