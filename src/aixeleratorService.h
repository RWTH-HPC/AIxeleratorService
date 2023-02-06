#ifndef AIXELERATORSERVICE_H_
#define AIXELERATORSERVICE_H_

#include "distributionStrategy/distributionStrategy.h"
#include "inferenceStrategy/inferenceStrategy.h"

#include <memory>

typedef enum AIFramework
{
    AIX_TORCH = 1,
    AIX_TENSORFLOW = 2,
} AIFramework;

class AIxeleratorService
{
    public:
        AIxeleratorService();

        void registerModel(std::string model_file);
        void registerTensors(
            std::vector<int64_t> input_shape, double* input_data,
            std::vector<int64_t> output_shape, double* output_data
        );
        void inference();
        void setBatchsize(int batchsize){ batchsize_ = batchsize; }

    private:
        std::string model_file_name_;
        std::vector<int64_t> input_shape_;
        std::vector<int64_t> output_shape_;
        double* input_data_;
        double* output_data_;
        int batchsize_;

        int my_rank_;
        AIFramework framework_;

        // make unique ptrs
        std::unique_ptr<InferenceStrategy> inferencing_;
        std::unique_ptr<DistributionStrategy> distributor_;
};

#endif