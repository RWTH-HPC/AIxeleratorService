#include "aixeleratorService/aixeleratorService_interfaceC.h"

#include <mpi.h>

#include <stdint.h>
#include <stdio.h>

int main( int argc, char *argv[] )
{
    MPI_Init(&argc, &argv);

    int my_rank = -1;
    int num_procs = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    printf("Test AIxeleratorService from C starting\n");
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

    printf("Creating AIxeleratorService object from C now!\n");
    printf("MPI Rank %d: registering input tensor for AIxeleratorService = (%g, %g)\n", my_rank, input_data[0], input_data[1]);
    AIxeleratorServiceHandle aixelerator = createAIxeleratorServiceDouble(model_file, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data, batch_size, MPI_COMM_WORLD);
    
    printf("MPI Rank %d: calling AIxeleratorService inference from C now!\n", my_rank);
    inferenceAIxeleratorServiceDouble(aixelerator);

    printf("MPI Rank %d: received output from AIxeleratorService from C = (%g, %g)\n", my_rank, output_data[0], output_data[1]);

    printf("MPI Rank %d: Deleting AIxeleratorService object from C now!\n", my_rank);
    deleteAIxeleratorServiceDouble(aixelerator);

    MPI_Finalize();

    printf("Test AIxeleratorService from C completed!\n");

    return 0;
}