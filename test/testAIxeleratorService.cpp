#include "aixeleratorService/aixeleratorService.h"

#include <mpi.h>

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Error: no model file found. Usage ./testAIxeleratorService <model_file_path>" << std::endl;
        return 1;
    }

    MPI_Init(&argc, &argv);

    int my_rank = -1;
    int num_procs = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::string model_file = argv[1];

    int n_samples = 2;
    
    std::vector<int64_t> input_shape = { n_samples, 2 };
    std::vector<double> input(n_samples*2, (double)my_rank);
    std::vector<int64_t> output_shape = { n_samples, 2 };
    std::vector<double> output(n_samples*2, -13.37);

    int batchsize = 1;

    std::cout << "MPI Rank " << my_rank << ": registering input tensor for AIxeleratorService = (" << input[0] << ", " << input[1] << ", " << input[2] << ", " << input[3] << ")" << std::endl;

    AIxeleratorService<double> aixelerator(
        model_file, 
        input_shape, input.data(), 
        output_shape, output.data(),
        batchsize, MPI_COMM_WORLD
    );

    std::cout << "MPI Rank " << my_rank << ": calling inference!" << std::endl;

    aixelerator.inference();

    std::cout << "MPI Rank " << my_rank << ": received output from AIxeleratorService = (" << output[0] << ", " << output[1] << ", " << output[2] << ", " << output[3] << ")" << std::endl;

    MPI_Finalize();
    return 0;
}
