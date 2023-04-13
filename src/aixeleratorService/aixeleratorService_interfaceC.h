#ifndef AIXELERATORSERVICE_INTERFACE_C_H_
#define AIXELERATORSERVICE_INTERFACE_C_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef void* AIxeleratorServiceHandle;
    AIxeleratorServiceHandle createAIxeleratorServiceDouble(
        char* model_file,
        int64_t* input_shape, int num_input_dims, double* input_data,
        int64_t* output_shape, int num_output_dims, double* output_data,
        int batchsize
    );
    AIxeleratorServiceHandle createAIxeleratorServiceFloat(
        char* model_file,
        int64_t* input_shape, int num_input_dims, float* input_data,
        int64_t* output_shape, int num_output_dims, float* output_data,
        int batchsize
    );

    void deleteAIxeleratorServiceDouble(AIxeleratorServiceHandle aixelerator);
    void deleteAIxeleratorServiceFloat(AIxeleratorServiceHandle aixelerator);
    
    void inferenceAIxeleratorServiceDouble(AIxeleratorServiceHandle aixelerator);
    void inferenceAIxeleratorServiceFloat(AIxeleratorServiceHandle aixelerator);


#ifdef __cplusplus
}
#endif

#endif