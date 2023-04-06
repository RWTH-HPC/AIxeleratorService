#ifndef AIXELERATORSERVICE_H_
#define AIXELERATORSERVICE_H_

#include "inferenceStrategy/inferenceStrategy.h"
#include "distributionStrategy/distributionStrategy.h"
#include "communicationStrategy/communicationStrategy.h"

#include <memory>

typedef enum AIFramework
{
    AIX_TORCH = 1,
    AIX_TENSORFLOW = 2,
    AIX_SOL = 3, 
    AIX_UNKNOWN
} AIFramework;

typedef enum InferenceMode
{
    AIX_CPU = 1,
    AIX_GPU = 2,
    AIX_HYBRID = 3 // NYI
} InferenceMode;

class AIxeleratorService
{
    public:
        AIxeleratorService() = delete;
        AIxeleratorService(
            std::string model_file,
            std::vector<int64_t> input_shape, double* input_data,
            std::vector<int64_t> output_shape, double* output_data,
            int batchsize
        );

        ~AIxeleratorService();

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
        std::vector<int64_t> input_shape_host_;
        std::vector<int64_t> input_shape_device_;
        
        std::vector<int64_t> output_shape_;
        std::vector<int64_t> output_shape_host_;
        std::vector<int64_t> output_shape_device_;

        double* input_data_;
        double* input_data_host_;
        double* input_data_device_;

        double* output_data_;
        double* output_data_host_;
        double* output_data_device_;
        int batchsize_; // TODO: remove this
        AIFramework framework_;
        InferenceMode inference_mode_;

        std::unique_ptr<DistributionStrategy> distributor_;
        std::unique_ptr<CommunicationStrategy> communicator_;
        std::unique_ptr<InferenceStrategy> inferencing_;

        std::unique_ptr<InferenceStrategy> inferencing_host_;
        std::unique_ptr<InferenceStrategy> inferencing_device_;

        std::unique_ptr<InferenceStrategy> createInferenceStrategy();
        std::pair<int64_t, int64_t> findHybridBatchsize();
        void initInferenceStrategy(std::pair<int64_t, int64_t> best_batchsizes);
};

#endif