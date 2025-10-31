#include "aixeleratorService/aixeleratorService.h"

#include <mpi.h>
#include <cstdlib>
#include <numeric>

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cerr << "Error: missing parameter. Usage: ./testAIxeleratorService <model_file_path> <total samples (cubes)> <num_GPUs> <host_fraction>" << std::endl;
        return 1;
    }

    MPI_Init(&argc, &argv);

    int my_rank = -1;
    int num_procs = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::string model_file = argv[1];

    char* endptr = nullptr;
    int64_t total_samples = std::strtoll(argv[2], &endptr, 10);
    std::cout << "Total number of samples (cubes) = " << total_samples << std::endl;

    int num_gpus = std::atoi(argv[3]);
    std::cout << "Number of GPUs for this experiment = " << num_gpus << std::endl;

    int hostfraction = 0.01 * std::atoi(argv[3]);
    std::cout << "Hybrid Host fraction for this experiment = " << num_gpus << std::endl;

    // number of cpus that will perform hybrid inference, so we exclude the specified number of GPUs
    // Moreover, we assume that on each node the same number of CPU processes is launched
    // e.g. num_gpus = 1 --> 95 cpus allocated on each CPU node (and 95+1=96) on the GPU node
    // e.g. num_gpus = 4 --> 92 cpus allocated on each CPU node (and 92+4=96) on the GPU node 
    // this way the number of cpus doubles if the number of allocated nodes is doubled --> easier to scale strong
    int num_cpus = num_procs - num_gpus;

    // samples per process
    int n_samples = total_samples / num_cpus;
    
    int inputSeqLen = 5;
    int cubeD = 8;
    int forecastwindow = 2; // make sure this is set according to he used model file of the TBL-Transformer!

    std::vector<int64_t> input_shape = { n_samples, inputSeqLen, cubeD * cubeD * cubeD};
    std::vector<int64_t> output_shape = { n_samples, forecastwindow, cubeD * cubeD * cubeD };

    int64_t num_input_elements = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<>());
    int64_t num_output_elements = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<>());

    std::vector<float> input(num_input_elements, (float)my_rank);
    std::vector<float> output(num_output_elements, -13.37);

    int batchsize = 65000; // empirically determined until crashes occur

    std::cout << "MPI Rank " << my_rank << ": n_samples = " << n_samples << ", setup input for AIxeleratorService!" << std::endl;

    bool isHybrid = true;
    float hostFraction = 0.01;

    AIxeleratorService<float> aixelerator(
        model_file, 
        input_shape, input.data(), 
        output_shape, output.data(),
        batchsize, MPI_COMM_WORLD,
        isHybrid, hostFraction
    );

    std::cout << "MPI Rank " << my_rank << ": calling inference!" << std::endl;

    aixelerator.inference();

    std::cout << "MPI Rank " << my_rank << ": received output from AIxeleratorService!" << std::endl;

    MPI_Finalize();
    return 0;
}
