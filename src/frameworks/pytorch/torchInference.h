#ifndef AIXSERVICETORCHINFERENCE
#define AIXSERVICETORCHINFERENCE

#include <torch/script.h>
#include <torch/torch.h>

#include <memory> // TODO: check if needed?
#include <mpi.h>

class torchInference 
{
private:
    torch::jit::script::Module torchModel_;                 //!< torch model
    int myDeviceNum_;

    torch::Tensor inputTensor_;  //!< tensor for model input on CPU
    torch::Tensor outputTensor_; //!< tensor for model output on CPU
    torch::Tensor inputTensorSlice_;  //!< tensor slice for model input on CPU
    torch::Tensor inputTensorGPU_;  //!< tensor for model input on GPU
    torch::Tensor outputTensorGPU_; //!< tensor for model output on GPU
    torch::TensorOptions options_{}; //!< tensor options
    unsigned int nCellsBatch_ = 0; //! number of cells per batch

 public:

    torchInference() = delete;  //!< delete default constructor
    //torchInference(const torchInference&) = delete; //< delete copy constructor
    //torchInference & operator=( const torchInference& ) = delete; //< delete copy assignment operator
    //torchInference( torchInference&& ) = delete; //< delete move constructor
    //torchInference& operator=( torchInference&& ) = delete; // delete move assignment operator
    
    torchInference(std::string modelFile, int deviceNum);

    ~torchInference();


    void allocateTensors(std::vector<int> &inputShape, std::vector<int> &outputShape);
    void forward(double* inputTensor, double* outputTensor, int batchsize);

 private:

    void batchedForward(int batchsize);
}; 

#endif // AIXSERVICETORCHINFERENCE