#include "distributionStrategy/roundRobinDistribution.h"
#include "communicationStrategy/collectiveCommunication.h"
#include "communicationStrategy/nonBlockingPtoPCommunication.h"

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

    std::vector<int64_t> input_shape = { 1, 2 };
    std::vector<double> input = { (double)my_rank, (double)my_rank };
    std::vector<int64_t> output_shape = { 1, 2 };
    std::vector<double> output = { -13.37, -13.37 };

    RoundRobinDistribution distributor(MPI_COMM_WORLD);
    /*
    CollectiveCommunication<double> communicator(
        input_shape, input.data(), 
        output_shape, output.data(), 
        distributor.isGPUController(), 
        *(distributor.getWorkGroupCommunicator())
    );
    */

    NonBlockingPtoPCommunication<double> communicator(
        input_shape, input.data(),
        output_shape, output.data(),
        0,
        *(distributor.getWorkGroupCommunicator())
    );

    std::cout << "Worker " << my_rank << " sends input: ";
    for (int j = 0; j < input.size(); j++)
    {
        std::cout << input[j] << ",";
    }
    std::cout << std::endl;    

    communicator.gatherInputData();
    
    if(distributor.isGPUController())
    {
        double* input_data_controller = communicator.getInputDataController();   
        int total_input_count = communicator.getTotalInputCount();

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
        double* input_data_controller = communicator.getInputDataController();
        int total_input_count = communicator.getTotalInputCount();  

        double* output_data_controller = communicator.getOutputDataController();   
        int total_output_count = communicator.getTotalOutputCount(); 

        // note: assuming input and output have same size
        std::memcpy(output_data_controller, input_data_controller, total_input_count * sizeof(double));
    }

    if(distributor.isGPUController())
    {
        double* output_data_controller = communicator.getOutputDataController();   
        int total_output_count = communicator.getTotalOutputCount();

        std::cout << "Controller will scatter output data: ";
        for (int j = 0; j < total_output_count; j++)
        {
            std::cout << output_data_controller[j] << ",";
        }
        std::cout << std::endl;
    }

    communicator.scatterOutputData();

    
    std::cout << "Worker " << my_rank << " recieved output: ";
    for (int j = 0; j < output.size(); j++)
    {
        std::cout << output[j] << ",";
    }
    std::cout << std::endl;    




    MPI_Finalize();
    return 0;
}
