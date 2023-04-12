#include <iostream>

#include "inferenceStrategy/torchInference/torchInference.h"

int main(int argc, char *argv[])
{
    std::cout << "Test for torchInference starting" << std::endl; 

    std::vector<int64_t> input_shape = {4, 2};
    //double input_data[] = { 1.0, 1.0 };
    //std::shared_ptr<double[]> p_input(input_data);
    double* input_data = new double[8] {    0.0, 0.0,
                                            1.0, 1.0,
                                            2.0, 2.0,
                                            3.0, 3.0
                                        };

    std::vector<int64_t> output_shape = {4, 2};
    //double output_data[] = { 0.0, 0.0 };
    //std::shared_ptr<double[]> p_output(output_data);
    double* output_data = new double[8] {   -13.37, -13.37,
                                            -13.37, -13.37,
                                            -13.37, -13.37,
                                            -13.37, -13.37,
                                        };

    int batchsize = 3;
    int device_id = 0;

    std::string model_file_name = "../models/torchModels/flexMLP-2x100x100x2.pt";

    TorchInference<double> torchInfer;
    //torchInfer.init(batchsize, device_id, model_file_name, input_shape, p_input, output_shape, p_output);
    torchInfer.init(batchsize, device_id, model_file_name, input_shape, input_data, output_shape, output_data);

    std::cout << "Torch Inference Test inference now!" << std::endl;

    torchInfer.inference();

    std::cout << "(" << input_data[0] << ", " << input_data[1] << ") --> (" << output_data[0] << ", " << output_data[1] << ")" << std::endl;
    std::cout << "(" << input_data[2] << ", " << input_data[3] << ") --> (" << output_data[2] << ", " << output_data[3] << ")" << std::endl;
    std::cout << "(" << input_data[4] << ", " << input_data[5] << ") --> (" << output_data[4] << ", " << output_data[5] << ")" << std::endl;
    std::cout << "(" << input_data[6] << ", " << input_data[7] << ") --> (" << output_data[6] << ", " << output_data[7] << ")" << std::endl;

    std::cout << "Torch Inference Test deallocating memory!" << std::endl;

    delete[] output_data;
    delete[] input_data;
    
    std::cout << "Torch Inference Test completed!" << std::endl;

    return 0;
}
