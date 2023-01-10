#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_TENSORFLOWINFERENCE_H_
#define AIXELERATORSERVICE_INFERENCESTRATEGY_TENSORFLOWINFERENCE_H_

#include "inferenceStrategy/inferenceStrategy.h"

#include <tensorflow/c/c_api.h>

class TensorflowInference : public InferenceStrategy
{
    public:
        TensorflowInference() = default;
        ~TensorflowInference() = default;

        void init();
        void inference() override;

    private:
        int batchsize_;

        TF_Session* session_;

        TF_Tensor* input_;
        TF_Tensor* output_;
};

#endif