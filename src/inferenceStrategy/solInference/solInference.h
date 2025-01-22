#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_SOLINFERENCE_H_
#define AIXELERATORSERVICE_INFERENCESTRATEGY_SOLINFERENCE_H_

#include "inferenceStrategy/inferenceStrategy.h"

#include <veda.h>

#include <vector>


class SOLInference : public InferenceStrategy
{
    public:
        SOLInference();
        ~SOLInference();

        void init(
            int batchsize, 
            int device_id, 
            std::string model_file_name, 
            std::vector<int64_t>& input_shape, double* inputData, 
            std::vector<int64_t>& output_shape, double* outputData
        ) override;

        void inference() override;

    private:
        int device_id_;
        std::string model_file_name_;
        int batchsize_;
        int num_batches_;
        int size_remaining_;

        VEDAcontext ctx_;
        VEDAmodule mod_;
        VEDAfunction func_;

        double* input_; // input ptr from application
        double* output_; // output ptr from application

        std::vector<int> tensor_offsets_;

        size_t input_batch_len_;
        VEDAdeviceptr input_batch_;
        size_t output_batch_len_;
        VEDAdeviceptr output_batch_;

        size_t input_remainder_len_;
        size_t output_remainder_len_;

};

#endif