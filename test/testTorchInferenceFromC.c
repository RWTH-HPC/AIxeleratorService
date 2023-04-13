#include "inferenceStrategy/torchInference/torchInference_interfaceC.h"

#include <stdint.h>
#include <stdio.h>

int main( int argc, char *argv[] )
{

    printf("Test for torchInference C interface starting\n");
    int num_input_dims = 2;
    int64_t input_shape[2] = {4, 2};
    double input_data[8] = {   
                                0.0, 0.0,
                                1.0, 1.0,
                                2.0, 2.0,
                                3.0, 3.0
                            };

    int num_output_dims = 2;
    int64_t output_shape[2] = {4, 2};
    double output_data[8] = {   
                                -13.37, -13.37,
                                -13.37, -13.37,
                                -13.37, -13.37,
                                -13.37, -13.37
                            };

    int batch_size = 3;
    int device_id = 0;

    char model_file[] = "../models/torchModels/flexMLP-2x100x100x2.pt";

    printf("Creating Torch Inference object now!\n");
    TorchInferenceHandle myTorch = createTorchInferenceDouble();

    printf("Init Torch Inference object now!\n");
    initTorchInferenceDouble(myTorch, batch_size, device_id, model_file, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data);
    
    printf("Torch Inference Test inference now!\n");
    forwardTorchInferenceDouble(myTorch);

    printf("(%g, %g) --> (%g, %g)\n", input_data[0], input_data[1], output_data[0], output_data[1]);
    printf("(%g, %g) --> (%g, %g)\n", input_data[2], input_data[3], output_data[2], output_data[3]);
    printf("(%g, %g) --> (%g, %g)\n", input_data[4], input_data[5], output_data[4], output_data[5]);
    printf("(%g, %g) --> (%g, %g)\n", input_data[4], input_data[7], output_data[6], output_data[7]);

    printf("Deleting Torch Inference object now!\n");
    deleteTorchInferenceDouble(myTorch);

    printf("Torch Inference Test completed!\n");

    return 0;
}