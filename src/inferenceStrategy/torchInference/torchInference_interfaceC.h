#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_TORCHINFERENCE_INTERFACE_C_H_
#define AIXELERATORSERVICE_INFERENCESTRATEGY_TORCHINFERENCE_INTERFACE_C_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef void* TorchInferenceHandle;

    TorchInferenceHandle createTorchInferenceDouble();
    TorchInferenceHandle createTorchInferenceFloat();

    void deleteTorchInferenceDouble(TorchInferenceHandle obj);
    void deleteTorchInferenceFloat(TorchInferenceHandle obj);

    void initTorchInferenceDouble(TorchInferenceHandle obj, int batchsize, int device_id, char* model_file, int64_t* input_shape, int num_input_dims, double* input_data, int64_t* output_shape, int num_output_dims, double* output_data);
    void initTorchInferenceFloat(TorchInferenceHandle obj, int batchsize, int device_id, char* model_file, int64_t* input_shape, int num_input_dims, float* input_data, int64_t* output_shape, int num_output_dims, float* output_data);

    void forwardTorchInferenceDouble(TorchInferenceHandle obj);
    void forwardTorchInferenceFloat(TorchInferenceHandle obj);

#ifdef __cplusplus
}
#endif

#endif