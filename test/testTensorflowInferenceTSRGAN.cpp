#include <iomanip>
#include <iostream>

#include "inferenceStrategy/tensorflowInference/tensorflowInference.h"

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);

    std::cout << "Test for tensorflowInference TSRGAN 3D starting" << std::endl; 

    /*
    const std::vector<int64_t> input_shape = {1, 2};
    double* input_data = new double[2] { 1.0, 1.0 };
    const std::vector<int64_t> output_shape = {1, 2};
    double* output_data = new double[2] { -13.37, -13.37 };
    */

    int upsampling = 4;

    int num_batches = 1;
    int num_channels = 3;
    std::vector<int64_t> input_shape = {num_batches, 2, 2, 2, num_channels};
    float* input_data = new float[24] { 
        0.1, 0.5, 0.2,
        0.15, 0.6, 0.1,

        0.4, 0.7, 0.3,
        0.8, 0.9, 0.2,

        0.8, 0.2, 0.45,
        0.35, 0.76, 0.98,

        0.23, 0.75, 0.17,
        0.82, 0.48, 0.89
    };

    std::vector<int64_t> output_shape = {num_batches, input_shape[1]*upsampling, input_shape[2]*upsampling, input_shape[3]*upsampling, num_channels};
    int out_size = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4];
    float* output_data = new float[out_size];
    for(int i = 0; i < out_size; i++)
    {
        output_data[i] = -13.37;
    }
    

    int batchsize = 1;
    int device_id = 0;

    std::string model_file_name;
    if(upsampling == 2)
    {
        model_file_name = "/work/rwth0792/fortran-ml-interface/model/tsrgan/TSRGAN_3D_36_2X_decay_gaussian.tf";
    }
    if(upsampling == 4)
    {
        model_file_name = "/work/rwth0792/fortran-ml-interface/model/tsrgan/TSRGAN_3D_36_4X_decay_gaussian.tf";   
    }

    TensorflowInference<float> tensorflowInfer;
    tensorflowInfer.setCommunicator(MPI_COMM_WORLD);
    tensorflowInfer.init(batchsize, device_id, model_file_name, input_shape, input_data, output_shape, output_data);

    std::cout << "Tensorflow Inference TSRGAN 3D Test inference now!" << std::endl;

    tensorflowInfer.inference();

    for(int i = 0; i < num_batches; i++){
        std::cout << "Batch " << i << std::endl;

        std::cout << "Input: " << std::endl;
        std::cout << "(";
        // channel dimension
        for(int k = 0; k < num_channels; k++){
            std::cout << "channel " << k << std::endl;
            // depth aka z-dimension
            for(int d = 0; d < 2; d++){
                // rows aka y-dim
                for(int r = 0; r < 2; r++){
                    // columns aka x-dim
                    for(int c = 0; c < 2; c++){
                        int idx = + d*8 + r*4 + c*2 + k;
                        std::cout << input_data[idx] << ", ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << ")" << std::endl;

        std::cout << "-->" << std::endl;

        /*
        std::cout << "Output:" << std::endl;
        std::cout << "(";
        for(int k = 0; k < num_channels; k++){
            std::cout << "channel " << k << std::endl;
            // depth aka z-dimension
            for(int d = 0; d < output_shape[3]; d++){
                // rows aka y-dim
                for(int r = 0; r < output_shape[2]; r++){
                    // columns aka x-dim
                    for(int c = 0; c < output_shape[1]; c++){
                        int idx = + d*64 + r*16 + c*4 + k;
                        std::cout << output_data[idx] << ", ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << ")" << std::endl;
        */
    }

    std::cout << "Output_data: " << std::endl;
    for(int i = 0; i < out_size; i++)
    {
        std::cout << std::setprecision(16) << output_data[i] << ",";
        if(i % num_channels == num_channels - 1)
        {
            std::cout << std::endl;
            if(i %(output_shape[1] * output_shape[2]) == (output_shape[1] * output_shape[2] - 1))
            {
                std::cout << std::endl;
            }
            if(i % output_shape[3] == output_shape[3] - 1)
            {
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;

    std::cout << "Tensorflow Inference TSRGAN 3D Test deallocating memory!" << std::endl;

    delete[] output_data;
    delete[] input_data;
    
    std::cout << "Tensorflow Inference TSRGAN 3D Test completed!" << std::endl;

    MPI_Finalize();

    return 0;
}
