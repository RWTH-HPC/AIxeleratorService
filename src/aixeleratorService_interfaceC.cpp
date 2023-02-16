#include "aixeleratorService_interfaceC.h"
#include "aixeleratorService.h"


#ifdef __cplusplus
extern "C" {
#endif

    AIxeleratorServiceHandle createAIxeleratorService(
        char* model_file,
        int64_t* input_shape, int num_input_dims, double* input_data,
        int64_t* output_shape, int num_output_dims, double* output_data,
        int batchsize
    ){
        std::string model_file_str(model_file);
        std::vector<int64_t> input_shape_vec(input_shape, input_shape + num_input_dims);
        std::vector<int64_t> output_shape_vec(output_shape, output_shape + num_output_dims);

        return (AIxeleratorService*) new AIxeleratorService(model_file_str, input_shape_vec, input_data, output_shape_vec, output_data, batchsize);
    }

    void deleteAIxeleratorService(AIxeleratorServiceHandle aixelerator)
    {
        delete (AIxeleratorService*) aixelerator;
    }

    void inferenceAIxeleratorService(AIxeleratorServiceHandle aixelerator)
    {
        ((AIxeleratorService*) aixelerator)->inference();    
    }

#ifdef __cplusplus
}
#endif
