#include <iostream>

#include "inferenceStrategy/tensorflowInference/tensorflowInference.h"

int main(int argc, char *argv[])
{
    std::cout << "Test for tensorflowInference starting" << std::endl; 

    /*
    const std::vector<int64_t> input_shape = {1, 2};
    double* input_data = new double[2] { 1.0, 1.0 };
    const std::vector<int64_t> output_shape = {1, 2};
    double* output_data = new double[2] { -13.37, -13.37 };
    */

    std::vector<int64_t> input_shape = {4, 2};
    double* input_data = new double[8] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0 };

    std::vector<int64_t> output_shape = {4, 2};
    double* output_data = new double[8] { -13.37, -13.37, -13.37, -13.37, -13.37, -13.37, -13.37, -13.37 };
    

    int batchsize = 3;
    int device_id = 0;

    std::string model_file_name = "../models/tensorflowModels/flexMLP-2x100x100x2.tf";

    TensorflowInference<double> tensorflowInfer;
    tensorflowInfer.init(batchsize, device_id, model_file_name, input_shape, input_data, output_shape, output_data);

    std::cout << "Tensorflow Inference Test inference now!" << std::endl;

    tensorflowInfer.inference();
    std::cout << "(" << input_data[0] << ", " << input_data[1] << ") --> (" << output_data[0] << ", " << output_data[1] << ")" << std::endl;
    std::cout << "(" << input_data[2] << ", " << input_data[3] << ") --> (" << output_data[2] << ", " << output_data[3] << ")" << std::endl;
    std::cout << "(" << input_data[4] << ", " << input_data[5] << ") --> (" << output_data[4] << ", " << output_data[5] << ")" << std::endl;
    std::cout << "(" << input_data[6] << ", " << input_data[7] << ") --> (" << output_data[6] << ", " << output_data[7] << ")" << std::endl;

    std::cout << "Tensorflow Inference Test deallocating memory!" << std::endl;

    delete[] output_data;
    delete[] input_data;
    
    std::cout << "Tensorflow Inference Test completed!" << std::endl;

    return 0;
}
