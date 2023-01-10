#include "inferenceStrategy/tensorflowInference.h"


void TensorflowInference::init()
{
    //session_ = TF_LoadSessionFromSavedModel()
}

void TensorflowInference::inference()
{
    /*
    int input_dims = TF_NumDims(input_);
    if ( input_dims < 2 )
    {
        std::cerr << "Error: input tensor for tensorflow inference needs at least two dimensions" << std::endl;
        std::terminate();
    }

    int batch_dim = TF_Dim(input_, 0);
    int second_dim = TF_Dim(input_, 1);
    int num_batches = batch_dim / batchsize_;
    int size_remaining = batch_dim % batchsize_;

    double* input_data = (double*) TF_TensorData(input_);
    double* output_data = (double*) TF_TensorData(output_);

    double* input_batch_data  = (double*) TF_TensorData(input_batch_);
    double* output_batch_data = (double*) TF_TensorData(output_batch_);

    for( int i = 0; i < num_batches; i++ )
    {
        // TODO: generalize for more than 2-dimensional tensors
        std::memcpy(input_batch_data, &(input_data)[i*batchsize_*second_dim], sizeof(double) * batch_dim * second_dim);

        TF_SessionRun(session_, )
    }
    */
}