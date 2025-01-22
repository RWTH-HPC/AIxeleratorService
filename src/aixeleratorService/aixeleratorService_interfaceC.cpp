#include "aixeleratorService/aixeleratorService_interfaceC.h"
#include "aixeleratorService/aixeleratorService.h"

#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

    AIxeleratorServiceHandle createAIxeleratorServiceDouble_F(
        char* model_file,
        int64_t* input_shape, int num_input_dims, double* input_data,
        int64_t* output_shape, int num_output_dims, double* output_data,
        int batchsize, int app_comm
    ){
        return createAIxeleratorServiceDouble(model_file, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data,  batchsize, MPI_Comm_f2c(app_comm));
    }

    AIxeleratorServiceHandle createAIxeleratorServiceFloat_F(
        char* model_file,
        int64_t* input_shape, int num_input_dims, float* input_data,
        int64_t* output_shape, int num_output_dims, float* output_data,
        int batchsize, int app_comm
    ){
        return createAIxeleratorServiceFloat(model_file, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data,  batchsize, MPI_Comm_f2c(app_comm));
    }


    AIxeleratorServiceHandle createAIxeleratorServiceDouble(
        char* model_file,
        int64_t* input_shape, int num_input_dims, double* input_data,
        int64_t* output_shape, int num_output_dims, double* output_data,
        int batchsize, MPI_Comm app_comm
    ){
        std::string model_file_str(model_file);
        std::vector<int64_t> input_shape_vec(input_shape, input_shape + num_input_dims);
        std::vector<int64_t> output_shape_vec(output_shape, output_shape + num_output_dims);

        return (AIxeleratorService<double>*) new AIxeleratorService<double>(model_file_str, input_shape_vec, input_data, output_shape_vec, output_data, batchsize, app_comm);
    }

    AIxeleratorServiceHandle createAIxeleratorServiceFloat(
        char* model_file,
        int64_t* input_shape, int num_input_dims, float* input_data,
        int64_t* output_shape, int num_output_dims, float* output_data,
        int batchsize, MPI_Comm app_comm
    ){
        std::string model_file_str(model_file);
        std::vector<int64_t> input_shape_vec(input_shape, input_shape + num_input_dims);
        std::vector<int64_t> output_shape_vec(output_shape, output_shape + num_output_dims);

        return (AIxeleratorService<float>*) new AIxeleratorService<float>(model_file_str, input_shape_vec, input_data, output_shape_vec, output_data, batchsize, app_comm);
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

    void setAIxeleratorServiceDebugTag(AIxeleratorServiceHandle aixelerator, char* debug_tag)
    {
        std::string tag(debug_tag);
        ((AIxeleratorService<float>*) aixelerator)->setDebugTag(tag);    
    }

#ifdef __cplusplus
}
#endif
