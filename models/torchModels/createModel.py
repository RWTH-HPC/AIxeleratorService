import torch
from helper import FlexMLP
import argparse
#import torchvision.models as models

def parseArguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('ninp', type=int,
                        help='Number of input neurons')
    parser.add_argument('nout', type=int,
                        help='Number of output neurons')
    parser.add_argument('-n', '--neurons', dest='neurons', nargs='+', default=[32, 32],
                        help='List of neurons in hidden layers, e.g. -n 32 64 32')
    parser.add_argument('--output', '-o', dest='model', required=False, default='Script.pt',
                        help='File name of save torch script, default: Script.pt')
    parser.add_argument('--with-sol', dest='withSol', default=False, action='store_true')
    parser.add_argument('--with-float', dest='withFloat', default=False, action='store_true')

    return parser.parse_args()

class TrainingModel(torch.nn.Module):
	def __init__(self, model):
		super().__init__()
		self.m_model = model
		self.m_loss  = torch.nn.L1Loss()
		
	def forward(self, x, y, z, target):
		output = self.m_model(x, y, z)
		loss = self.m_loss(A, target)
		return sol.no_grad(output), loss

def main():
    args = parseArguments()

    n_inp = args.ninp
    n_out = args.nout
    n_neurons = [int(x) for x in args.neurons]

    if args.withFloat:
        data_type = torch.float
        m = FlexMLP(n_inp, n_out, n_neurons).requires_grad_(False)
    else:
        data_type = torch.double
        m = FlexMLP(n_inp, n_out, n_neurons).double().requires_grad_(False)

    if args.withSol:
        
        import sol
       
        # create dummy input
        nCells = 224 * 224 * 108
        nFields = 2

        input = torch.ones(1, nFields, dtype=data_type)
        target = torch.ones(1, nFields, dtype=data_type)

        #py_model = models.__dict__["alexnet"]()
        #input_alex = torch.rand(32, 3, 224, 224)
        #target_alex = torch.rand(32, 1000)
        #opt = sol.optimize(py_model, input_alex, target_alex)
        #py_model.load_state_dict(opt.state_dict(), strict=False)
        #sol.deploy(py_model, sol.input())



        #opt = sol.optimize(m, input, target)

        #sol.devices()

        #m.load_state_dict(opt.state_dict(), strict=False)
        sol.deploy(m, "shared_lib", {"lib_name": "libFlexMLP", "func_name": "forward", "path": "./libFlexMLP", "device_type": "ve"}, input)

    s = torch.jit.script(m)
    s.save(args.model)

    # TODO: refactor into own script to test the model!
    #loaded_model = torch.jit.load('flexMLP-2x100x100x2.pt')
    #loaded_model.eval()
    #test_input = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=torch.float64)
    #test_output = loaded_model(test_input)
    #print(test_output)

    #test_input = torch.tensor([[0.0, 1.0], [2.0, 3.0], [0.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
    #test_output = loaded_model(test_input)
    #print(test_output)

if __name__ == "__main__":
    main()
