#include "aixeleratorService/aixeleratorService_interfaceC.h"
#include "aixeleratorService/aixeleratorService.h"


#ifdef __cplusplus
extern "C" {
#endif

    AIxeleratorServiceHandle createAIxeleratorServiceDouble(
        char* model_file,
        int64_t* input_shape, int num_input_dims, double* input_data,
        int64_t* output_shape, int num_output_dims, double* output_data,
        int batchsize
    ){
        std::string model_file_str(model_file);
        std::vector<int64_t> input_shape_vec(input_shape, input_shape + num_input_dims);
        std::vector<int64_t> output_shape_vec(output_shape, output_shape + num_output_dims);

        return (AIxeleratorService<double>*) new AIxeleratorService<double>(model_file_str, input_shape_vec, input_data, output_shape_vec, output_data, batchsize);
    }

    AIxeleratorServiceHandle createAIxeleratorServiceFloat(
        char* model_file,
        int64_t* input_shape, int num_input_dims, float* input_data,
        int64_t* output_shape, int num_output_dims, float* output_data,
        int batchsize
    ){
        std::string model_file_str(model_file);
        std::vector<int64_t> input_shape_vec(input_shape, input_shape + num_input_dims);
        std::vector<int64_t> output_shape_vec(output_shape, output_shape + num_output_dims);

        return (AIxeleratorService<float>*) new AIxeleratorService<float>(model_file_str, input_shape_vec, input_data, output_shape_vec, output_data, batchsize);
    }

    void deleteAIxeleratorServiceDouble(AIxeleratorServiceHandle aixelerator)
    {
        delete (AIxeleratorService<double>*) aixelerator;
    }

    void deleteAIxeleratorServiceFloat(AIxeleratorServiceHandle aixelerator)
    {
        delete (AIxeleratorService<float>*) aixelerator;
    }

    void inferenceAIxeleratorServiceDouble(AIxeleratorServiceHandle aixelerator)
    {
        ((AIxeleratorService<double>*) aixelerator)->inference();    
    }

    void inferenceAIxeleratorServiceFloat(AIxeleratorServiceHandle aixelerator)
    {
        ((AIxeleratorService<float>*) aixelerator)->inference();    
    }

#ifdef __cplusplus
}
#endif
