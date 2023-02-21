#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_TORCHINFERENCE_INTERFACE_C_H_
#define AIXELERATORSERVICE_INFERENCESTRATEGY_TORCHINFERENCE_INTERFACE_C_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef void* TorchInferenceHandle;
    TorchInferenceHandle createTorchInference();
    void deleteTorchInference(TorchInferenceHandle obj);
    void initTorchInference(TorchInferenceHandle obj, int batchsize, int device_id, char* model_file, int64_t* input_shape, int num_input_dims, double* input_data, int64_t* output_shape, int num_output_dims, double* output_data);
    void forwardTorchInference(TorchInferenceHandle obj);

#ifdef __cplusplus
}
#endif

#endif