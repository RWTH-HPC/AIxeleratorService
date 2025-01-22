import argparse
import torch

def parseArguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', dest='model', required=True, help='File name of saved torch script model')

    return parser.parse_args()

def inferModel(model):
    loaded_model = torch.jit.load(model)
    loaded_model.eval()
    test_input = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=torch.float64)
    test_output = loaded_model(test_input)
    print(test_output)

    test_input = torch.tensor([[0.0, 1.0], [2.0, 3.0], [0.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
    test_output = loaded_model(test_input)
    print(test_output)

def main():
    args = parseArguments()
    model = args.model
    inferModel(model)

if __name__ == "__main__":
    main()