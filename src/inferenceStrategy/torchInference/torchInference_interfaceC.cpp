#include "torchInference_interfaceC.h"
#include "torchInference.h"

#ifdef __cplusplus
extern "C" {
#endif

    TorchInferenceHandle createTorchInference()
    {
        return (TorchInference*) new TorchInference();
    }

    void deleteTorchInference(TorchInferenceHandle obj)
    {
        delete (TorchInference*) obj;
    }

    void initTorchInference(TorchInferenceHandle obj, int batchsize, int device_id, char* model_file, int64_t* input_shape, int num_input_dims, double* input_data, int64_t* output_shape, int num_output_dims, double* output_data)
    {
        std::string model_file_name(model_file);
        std::vector<int64_t> input_shape_vec(input_shape, input_shape + num_input_dims);
        std::vector<int64_t> output_shape_vec(output_shape, output_shape + num_output_dims);

        ((TorchInference*) obj)->init(batchsize, device_id, model_file_name, input_shape_vec, input_data, output_shape_vec, output_data);
    }

    void forwardTorchInference(TorchInferenceHandle obj)
    {
       ((TorchInference*) obj)->inference();
    }

#ifdef __cplusplus
}
#endif