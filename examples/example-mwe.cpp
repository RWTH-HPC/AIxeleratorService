#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>
#include <aixeleratorService.h>


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    std::vector<int> inputShape{3, 2};
    std::vector<int> outputShape = inputShape;

    int totalDataCount = 0;
    for (const auto& value: inputShape)
    {
        totalDataCount += value;
    }

    double* input = (double*) malloc(totalDataCount * sizeof(double));   
    double* output = (double*) malloc(totalDataCount * sizeof(double));

    // init input and output
    for(int i = 0; i < totalDataCount; i++)
    {
        input[i] = 1.0;
        output[i] = 0.0;
    }

    std::string modelFile = "Script.pt";
    AIFramework framework = AIX_TORCH;
    AIxeleratorService aixService(framework, modelFile);

    aixService.registerTensorShape(inputShape, outputShape);
    // inference: input -> ML model -> output
    aixService.inference(input, totalDataCount, output, totalDataCount);

    // print ML output
    std::cout << "ML output: ";
    for(int i = 0; i < totalDataCount; i++)
    {
        std::cout << output[i] << " ,";
    }
    std::cout << std::endl;


    MPI_Finalize();
    return 0;
}