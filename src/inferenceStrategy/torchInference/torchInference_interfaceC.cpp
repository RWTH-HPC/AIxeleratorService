#include "inferenceStrategy/torchInference/torchInference_interfaceC.h"
#include "inferenceStrategy/torchInference/torchInference.h"

#ifdef __cplusplus
extern "C" {
#endif

    TorchInferenceHandle createTorchInferenceDouble()
    {
        return (TorchInference<double>*) new TorchInference<double>();
    }

    TorchInferenceHandle createTorchInferenceFloat()
    {
        return (TorchInference<float>*) new TorchInference<float>();
    }

    void deleteTorchInferenceDouble(TorchInferenceHandle obj)
    {
        delete (TorchInference<double>*) obj;
    }

    void deleteTorchInferenceFloat(TorchInferenceHandle obj)
    {
        delete (TorchInference<float>*) obj;
    }

    void initTorchInferenceDouble(TorchInferenceHandle obj, int batchsize, int device_id, char* model_file, int64_t* input_shape, int num_input_dims, double* input_data, int64_t* output_shape, int num_output_dims, double* output_data)
    {
        std::string model_file_name(model_file);
        std::vector<int64_t> input_shape_vec(input_shape, input_shape + num_input_dims);
        std::vector<int64_t> output_shape_vec(output_shape, output_shape + num_output_dims);

        ((TorchInference<double>*) obj)->init(batchsize, device_id, model_file_name, input_shape_vec, input_data, output_shape_vec, output_data);
    }

    void initTorchInferenceFloat(TorchInferenceHandle obj, int batchsize, int device_id, char* model_file, int64_t* input_shape, int num_input_dims, float* input_data, int64_t* output_shape, int num_output_dims, float* output_data)
    {
        std::string model_file_name(model_file);
        std::vector<int64_t> input_shape_vec(input_shape, input_shape + num_input_dims);
        std::vector<int64_t> output_shape_vec(output_shape, output_shape + num_output_dims);

        ((TorchInference<float>*) obj)->init(batchsize, device_id, model_file_name, input_shape_vec, input_data, output_shape_vec, output_data);
    }

    void forwardTorchInferenceDouble(TorchInferenceHandle obj)
    {
       ((TorchInference<double>*) obj)->inference();
    }

    void forwardTorchInferenceFloat(TorchInferenceHandle obj)
    {
       ((TorchInference<float>*) obj)->inference();
    }

#ifdef __cplusplus
}
#endif