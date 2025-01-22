#include <iostream>
#include <vector>
#include <mpi.h>

#include "inferenceStrategy/tensorflowInference/tensorflowInference.h"

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);

    std::cout << "Test for tensorflowInferenceConvolution 3D multichannel starting" << std::endl; 

    /*
    const std::vector<int64_t> input_shape = {1, 2};
    double* input_data = new double[2] { 1.0, 1.0 };
    const std::vector<int64_t> output_shape = {1, 2};
    double* output_data = new double[2] { -13.37, -13.37 };
    */

    int num_batches = 1;
    int num_channels = 3;
    std::vector<int64_t> input_shape = {num_batches, 3, 3, 3, num_channels};
    int elem_per_batch_in = 27;
    int elem_per_slice_in = 9;
    float* input_data = new float[81] { 
          1.0,  28.0,  55.0,
          2.0,  29.0,  56.0,
          3.0,  30.0,  57.0,

          4.0, 31.0, 58.0,
         5.0, 32.0, 59.0,
         6.0, 33.0, 60.0,

         7.0, 34.0, 61.0,
         8.0, 35.0, 62.0,
         9.0, 36.0, 63.0,

        10.0, 37.0, 64.0,
        11.0, 38.0, 65.0,
        12.0, 39.0, 66.0,

        13.0, 40.0, 67.0,
        14.0, 41.0, 68.0,
        15.0, 42.0, 69.0,

        16.0, 43.0, 70.0,
        17.0, 44.0, 71.0,
        18.0, 45.0, 72.0,

        19.0, 46.0, 73.0,
        20.0, 47.0, 74.0,
        21.0, 48.0, 75.0,

        22.0, 49.0, 76.0,
        23.0, 50.0, 77.0,
        24.0, 51.0, 78.0,

        25.0, 52.0, 79.0,
        26.0, 53.0, 80.0,
        27.0, 54.0, 81.0
    };

    std::vector<int64_t> output_shape = {num_batches, 2, 2, 2, 1};
    int elem_per_batch_out = 8;
    int elem_per_slice_out = 4;
    float* output_data = new float[8] { 
        -13.37, -13.37, 
        -13.37, -13.37,
        
        -13.37, -13.37, 
        -13.37, -13.37,
    };
    

    int batchsize = 1;
    int device_id = 0;

    std::string model_file_name = "/work/rwth0792/fortran-ml-interface/model/cnn3d-multichannel-test/testConvolution3D-multichannel.tf";

    TensorflowInference<float> tensorflowInfer;
    tensorflowInfer.setCommunicator(MPI_COMM_WORLD);
    tensorflowInfer.init(batchsize, device_id, model_file_name, input_shape, input_data, output_shape, output_data);

    std::cout << "Tensorflow Inference Convolution 3D multichannel test inference now!" << std::endl;

    tensorflowInfer.inference();

    for(int i = 0; i < num_batches; i++){
        std::cout << "Batch " << i << std::endl;

        std::cout << "Input: " << std::endl;
        std::cout << "(";
        // channel dimension
        for(int k = 0; k < num_channels; k++){
            std::cout << "channel " << k << std::endl;
            // depth aka z-dimension
            for(int d = 0; d < 3; d++){
                // rows aka y-dim
                for(int r = 0; r < 3; r++){
                    // columns aka x-dim
                    for(int c = 0; c < 3; c++){
                        int idx = + d*27 + r*9 + c*3 + k;
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

        std::cout << "Output:" << std::endl;
        std::cout << "(";
        for(int j = 0; j < 2; j++){
            int idx_out = i*elem_per_batch_out+j*elem_per_slice_out;

            std::cout << output_data[idx_out+0] << ", " << output_data[idx_out+1] << ", " << std::endl;
            std::cout << " " << output_data[idx_out+2] << ", " << output_data[idx_out+3] << "," << std::endl;
        }
        std::cout << ")" << std::endl;
    }

    std::cout << "Output_data: " << std::endl;
    for(int i = 0; i < 8; i++)
    {
        std::cout << output_data[i] << ",";
    }
    std::cout << std::endl;

    std::cout << "Tensorflow Inference Convolution 3D Test deallocating memory!" << std::endl;

    delete[] output_data;
    delete[] input_data;
    
    std::cout << "Tensorflow Inference Convolution 3D Test completed!" << std::endl;

    MPI_Finalize();

    return 0;
}
