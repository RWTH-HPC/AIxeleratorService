#include <iostream>

#include "inferenceStrategy/solInference/solInference.h"

#include <memory>


int main(int argc, char *argv[])
{
    vedaInit(0);

    std::cout << "Test for SOLInference starting" << std::endl; 

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

    int num_devices = 0;
    vedaDeviceGetCount(&num_devices);
    std::cout << "vedaDeviceGetCount = " << num_devices << std::endl;

    std::string model_file_name = "../models/solModels/libFlexMLP_3x2x100x100x2/wrapper/BUILD/liblibFlexMLP_3x2x100x100x2_veda_wrapper.vso";

    SOLInference* solInfer = new SOLInference(); // heap object so we can delete it before calling vedaExit()

    solInfer->init(batchsize, device_id, model_file_name, input_shape, input_data, output_shape, output_data);

    std::cout << "SOL Inference Test inference now!" << std::endl;

    solInfer->inference();

    std::cout << "(" << input_data[0] << ", " << input_data[1] << ") --> (" << output_data[0] << ", " << output_data[1] << ")" << std::endl;
    std::cout << "(" << input_data[2] << ", " << input_data[3] << ") --> (" << output_data[2] << ", " << output_data[3] << ")" << std::endl;
    std::cout << "(" << input_data[4] << ", " << input_data[5] << ") --> (" << output_data[4] << ", " << output_data[5] << ")" << std::endl;
    std::cout << "(" << input_data[6] << ", " << input_data[7] << ") --> (" << output_data[6] << ", " << output_data[7] << ")" << std::endl;

    std::cout << "SOL Inference Test deallocating memory!" << std::endl;

    delete[] output_data;
    delete[] input_data;
    
    std::cout << "SOL Inference Test completed!" << std::endl;

    delete solInfer; // destructor of SOLInference performs some VEDA cleanup functions and should be called before vedaExit()
    vedaExit();

    return 0;
}
