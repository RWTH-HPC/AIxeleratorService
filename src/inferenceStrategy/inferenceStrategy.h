#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_H_
#define AIXELERATORSERVICE_INFERENCESTRATEGY_H_

#include <string>
#include <vector>

template<typename T>
class InferenceStrategy
{
    public:
        virtual ~InferenceStrategy() = default;

        //virtual void setInput(std::vector<int> input_shape, double* input_data) = 0;
        //virtual void setOutput(std::vector<int> output_shape, double* output_data) = 0;
        virtual void init(
            int batchsize, 
            int device_id, 
            std::string model_file_name, 
            std::vector<int64_t>& input_shape, T* inputData, 
            std::vector<int64_t>& output_shape, T* outputData
        ) = 0;
        virtual void inference() = 0;
};

#endif