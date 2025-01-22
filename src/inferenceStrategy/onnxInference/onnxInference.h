//
// Created by co007276 on 8/4/23.
//

#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_ONNXINFERENCE_H
#define AIXELERATORSERVICE_INFERENCESTRATEGY_ONNXINFERENCE_H

#include "inferenceStrategy/inferenceStrategy.h"
#include "onnxruntime_cxx_api.h"

template<typename T>
class ONNXInference : public InferenceStrategy<T>
{
    private:
        Ort::Env env;
        Ort::SessionOptions session_options;
        int64_t batchsize_;
        int num_batches_;
        int device_id_;
        std::vector<std::vector<Ort::Value>> input_;
        //std::vector<Ort::Value> output_;
        T* output_;
        Ort::Session session_;
        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;
        int64_t batch_dim_;
    public:
        ONNXInference(std::string& model_file_name):
        env(ORT_LOGGING_LEVEL_WARNING, "onnx-model-batch-inference"),
        session_(env, model_file_name.c_str(), session_options){}
        ~ONNXInference() = default;

        void init(
            int batchsize,
            int device_id,
            std::string model_file_name,
            std::vector<int64_t>& input_shape, T* inputData,
            std::vector<int64_t>& output_shape, T* outputData

        ) override;
        void inference() override;
};

#endif //AIXELERATORSERVICE_INFERENCESTRATEGY_ONNXINFERENCE_H
