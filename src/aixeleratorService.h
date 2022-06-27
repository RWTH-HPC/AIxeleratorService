#ifndef AIXSERVICE
#define AIXSERVICE

//#include "./models/AIModel.h"
#include "./frameworks/pytorch/torchInference.h"

#include <string>
#include <mpi.h>

typedef enum AIFramework
{
    AIX_TORCH = 1,
    AIX_TENSORFLOW = 2,
} AIFramework;

class AIxeleratorService
{
    private:
        MPI_Comm workGroupComm_;
        bool isGPUMaster_;
        int myGPUDevice_;
        AIFramework framework_;
        int nRanksWorkGroup_;

        // only allocated on GPUMaster ranks
        int inputSizeTotal_;
        int outputSizeTotal_;
        int* inputCounts_;
        int* inputDisplacements_;
        int* outputCounts_;
        int* outputDisplacements_;
        double* inputTensorData_;
        double* outputTensorData_;

        torchInference torchInf_;

        int deviceCount();
        void initWorkgroup(MPI_Comm& workGroupComm);
        void gatherTensorData(double* input, int inputCount);
        void scatterTensorData(double* output, int outputCount);

    public:
        AIxeleratorService(AIFramework framework, std::string modelFile);
        ~AIxeleratorService();

        void registerTensorShape(std::vector<int> &inputShape, std::vector<int> &outputShape);
        void inference(double* input, int inputCount, double* output, int outputCount);

};

#endif // AIXSERVICE