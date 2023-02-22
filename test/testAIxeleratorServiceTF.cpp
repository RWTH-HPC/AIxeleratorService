#include "aixeleratorService/aixeleratorService.h"

#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int my_rank = -1;
    int num_procs = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::string model_file = "../models/tensorflowModels/flexMLP-2x100x100x2.tf";

    std::vector<int64_t> input_shape = { 1, 2 };
    std::vector<double> input = { (double)my_rank, (double)my_rank };

    std::vector<int64_t> output_shape = { 1, 2 };
    std::vector<double> output = { -13.37, -13.37 };

    int batchsize = 3;

    std::cout << "MPI Rank " << my_rank << ": registering input tensor for AIxeleratorService = (" << input[0] << ", " << input[1] << ")" << std::endl;

    AIxeleratorService aixelerator(
        model_file,
        input_shape, input.data(),
        output_shape, output.data(),
        batchsize
    );
    
    std::cout << "MPI Rank " << my_rank << ": calling inference!" << std::endl;

    aixelerator.inference();

    std::cout << "MPI Rank " << my_rank << ": received output from AIxeleratorService = (" << output[0] << ", " << output[1] << ")" << std::endl;

    MPI_Finalize();
    return 0;
}
