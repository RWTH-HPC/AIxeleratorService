import torch
import argparse
import onnx

# Invoke like this:
# conda activate onnx_env
# python3 onnxModels/convertFromTorch.py -mip torchModels/flexMLP-2x100x100x2.pt -mop onnxModels/flexMLP-1x100x100x1.onnx -ninp 1 -nout 1
# OR for float dtype: 
# python3 onnxModels/convertFromTorch.py -mip torchModels/flexMLP-float-2x100x100x2.pt -mop onnxModels/flexMLP-float-1x100x100x1.onnx -ninp 1 -nout 1 --with-float
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_input_path', '-mip', type=str, help='Path to torch model .pt file')
    parser.add_argument('--model_output_path', '-mop', type=str, help='Path where model is saved in ONNX format')
    parser.add_argument('--n_inputs', '-ninp', type=int,
                        help='Number of input neurons')
    parser.add_argument('--n_outputs', '-nout', type=int,
                        help='Number of output neurons')
    parser.add_argument('--with-float', dest='withFloat', default=False, action='store_true')
    args = parser.parse_args()

    model = torch.jit.load(args.model_input_path)
    # set model to eval mode
    model.eval()
    if args.withFloat:
        data_type = torch.float
    else:
        data_type = torch.double
    dummy_input = torch.randn(args.n_inputs, 2, dtype=data_type) # ReLu and double dtype leads to error: https://github.com/microsoft/onnxruntime/issues/6320
    input_names = ["input"] #+ ["input_%d" % i for i in range(args.n_inputs)]  # Add one input name for batch_size
    output_names = ["output"]  #+ ["output_%d" % i for i in range(args.n_outputs)]  # Add one output name for batch_size
    torch.onnx.export(model, dummy_input, args.model_output_path, export_params=True, opset_version=10,
                      do_constant_folding=True, input_names=input_names, output_names=output_names,
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})


if __name__ == "__main__":
    main()