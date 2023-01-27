#include "distributionStrategy/roundRobinDistribution.h"

#include <iostream>
#include <vector>
#include <mpi.h>
#include <numeric>
#include <cstring>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int my_rank = -1;
    int num_procs = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<double> input = { (double)my_rank, (double)my_rank };
    std::vector<double> output = { -13.37, -13.37 };

    RoundRobinDistribution distributor(input.size(), input.data(), output.size(), output.data());

    std::cout << "Worker " << my_rank << " sends input: ";
    for (int j = 0; j < input.size(); j++)
    {
        std::cout << input[j] << ",";
    }
    std::cout << std::endl;    

    distributor.gatherInputData();
    
    if(distributor.isGPUController())
    {
        double* input_data_controller = distributor.getInputDataController();   
        int total_input_count = distributor.getTotalInputCount();

        std::cout << "Controller gathered input data: ";
        for (int j = 0; j < total_input_count; j++)
        {
            std::cout << input_data_controller[j] << ",";
        }
        std::cout << std::endl;
    }

    // in real use case inference will happen here
    if(distributor.isGPUController())
    {
        double* input_data_controller = distributor.getInputDataController();
        int total_input_count = distributor.getTotalInputCount();  

        double* output_data_controller = distributor.getOutputDataController();   
        int total_output_count = distributor.getTotalOutputCount(); 

        // note: assuming input and output have same size
        std::memcpy(output_data_controller, input_data_controller, total_input_count * sizeof(double));
    }

    if(distributor.isGPUController())
    {
        double* output_data_controller = distributor.getOutputDataController();   
        int total_output_count = distributor.getTotalOutputCount();

        std::cout << "Controller will scatter output data: ";
        for (int j = 0; j < total_output_count; j++)
        {
            std::cout << output_data_controller[j] << ",";
        }
        std::cout << std::endl;
    }

    distributor.scatterOutputData();

    
    std::cout << "Worker " << my_rank << " recieved output: ";
    for (int j = 0; j < output.size(); j++)
    {
        std::cout << output[j] << ",";
    }
    std::cout << std::endl;    




    MPI_Finalize();
    return 0;
}
