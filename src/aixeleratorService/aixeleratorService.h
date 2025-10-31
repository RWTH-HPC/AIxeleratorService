#ifndef AIXELERATORSERVICE_H_
#define AIXELERATORSERVICE_H_

#include "inferenceStrategy/inferenceStrategy.h"
#include "distributionStrategy/distributionStrategy.h"
#include "communicationStrategy/communicationStrategy.h"

#include <memory>
#include <optional>

typedef enum AIFramework
{
    AIX_TORCH = 1,
    AIX_TENSORFLOW = 2,
    AIX_ONNX = 3,
    AIX_SOL = 4, 
    AIX_UNKNOWN
} AIFramework;

template<typename T>
class AIxeleratorService
{
    public:
        AIxeleratorService() = delete;
        AIxeleratorService(
            std::string model_file,
            std::vector<int64_t> input_shape, T* input_data,
            std::vector<int64_t> output_shape, T* output_data,
            int batchsize, MPI_Comm app_comm,
            bool enable_hybrid = false,
            std::optional<float> host_fraction = std::nullopt
        );

        ~AIxeleratorService();

        void registerModel(std::string model_file);
        void inference();
        void setBatchsize(int batchsize){ batchsize_ = batchsize; }

        void setDebugTag(std::string tag){ debug_tag_ = tag; }

    private:
        int my_rank_;

        std::string model_file_name_;
        std::string debug_tag_;

        std::vector<int64_t> input_shape_;
        std::vector<int64_t> input_shape_host_;
        std::vector<int64_t> input_shape_device_;
        
        std::vector<int64_t> output_shape_;
        std::vector<int64_t> output_shape_host_;
        std::vector<int64_t> output_shape_device_;

        T* input_data_;
        T* input_data_host_;
        T* input_data_device_;

        T* output_data_;
        T* output_data_host_;
        T* output_data_device_;
        int batchsize_; // TODO: remove this
        bool enable_hybrid_;
        std::optional<float> host_fraction_;

        AIFramework framework_;

        std::unique_ptr<DistributionStrategy> distributor_;
        std::unique_ptr<CommunicationStrategy<T>> communicator_;
        std::unique_ptr<InferenceStrategy<T>> inferencing_;

        std::unique_ptr<InferenceStrategy<T>> inferencing_host_;
        std::unique_ptr<InferenceStrategy<T>> inferencing_device_;

        std::unique_ptr<InferenceStrategy<T>> createInferenceStrategy();
        void initInferenceStrategy(std::pair<int64_t, int64_t> best_batchsizes);
};

#endif