#include <iostream>

#include "inferenceStrategy/tensorflowInference/tensorflowInference.h"

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);

    std::cout << "Test for tensorflowInferenceConvolution 3D starting" << std::endl; 

    /*
    const std::vector<int64_t> input_shape = {1, 2};
    double* input_data = new double[2] { 1.0, 1.0 };
    const std::vector<int64_t> output_shape = {1, 2};
    double* output_data = new double[2] { -13.37, -13.37 };
    */

    int num_batches = 3;
    std::vector<int64_t> input_shape = {num_batches, 3, 3, 3, 1};
    int elem_per_batch_in = 27;
    int elem_per_slice_in = 9;
    float* input_data = new float[81] { 
         1.0,  2.0,  3.0,
         4.0,  5.0,  6.0,
         7.0,  8.0,  9.0,

        10.0, 11.0, 12.0,
        13.0, 14.0, 15.0,
        16.0, 17.0, 18.0,

        19.0, 20.0, 21.0,
        22.0, 23.0, 24.0,
        25.0, 26.0, 27.0,


         1.0,  2.0,  3.0,
         4.0,  5.0,  6.0,
         7.0,  8.0,  9.0,

        10.0, 11.0, 12.0,
        13.0, 14.0, 15.0,
        16.0, 17.0, 18.0,

        19.0, 20.0, 21.0,
        22.0, 23.0, 24.0,
        25.0, 26.0, 27.0,


         1.0,  2.0,  3.0,
         4.0,  5.0,  6.0,
         7.0,  8.0,  9.0,

        10.0, 11.0, 12.0,
        13.0, 14.0, 15.0,
        16.0, 17.0, 18.0,

        19.0, 20.0, 21.0,
        22.0, 23.0, 24.0,
        25.0, 26.0, 27.0
    };

    std::vector<int64_t> output_shape = {num_batches, 2, 2, 2, 1};
    int elem_per_batch_out = 8;
    int elem_per_slice_out = 4;
    float* output_data = new float[24] { 
        -13.37, -13.37, 
        -13.37, -13.37,
        
        -13.37, -13.37, 
        -13.37, -13.37,


        -13.37, -13.37, 
        -13.37, -13.37,
        
        -13.37, -13.37, 
        -13.37, -13.37,


        -13.37, -13.37, 
        -13.37, -13.37,
        
        -13.37, -13.37, 
        -13.37, -13.37
    };
    

    int batchsize = 2;
    int device_id = 0;

    std::string model_file_name = "../models/tensorflowModels/testConvolution3D.tf";

    TensorflowInference<float> tensorflowInfer;
    tensorflowInfer.setCommunicator(MPI_COMM_WORLD);
    tensorflowInfer.init(batchsize, device_id, model_file_name, input_shape, input_data, output_shape, output_data);

    std::cout << "Tensorflow Inference Convolution 3D Test inference now!" << std::endl;

    tensorflowInfer.inference();

    for(int i = 0; i < num_batches; i++){
        std::cout << "Batch " << i << std::endl;

        std::cout << "Input: " << std::endl;
        std::cout << "(";
        for(int j = 0; j < 3; j++){
            int idx_in = i*elem_per_batch_in+j*elem_per_slice_in;
    
            std::cout << input_data[idx_in+0] << ", " << input_data[idx_in+1] << ", " << input_data[idx_in+2] << ", " << std::endl;
            std::cout << " " << input_data[idx_in+3] << ", " << input_data[idx_in+4] << ", " << input_data[idx_in+5] << ", " << std::endl;
            std::cout << " " << input_data[idx_in+6] << ", " << input_data[idx_in+7] << ", " << input_data[idx_in+8] << "," << std::endl;

            if(j < 2){
                std::cout << std::endl;
            }
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

    std::cout << "Tensorflow Inference Convolution 3D Test deallocating memory!" << std::endl;

    delete[] output_data;
    delete[] input_data;
    
    std::cout << "Tensorflow Inference Convolution 3D Test completed!" << std::endl;

    MPI_Finalize();

    return 0;
}
