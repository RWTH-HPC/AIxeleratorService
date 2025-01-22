import torch
import torch.nn as nn
from typing import List

class FlexMLP(torch.nn.Module):
    def __init__(self, n_inp: int, n_out: int, n_hidden_neurons: List[int] =[32, 32], activation_fn: torch.nn.Module = nn.ReLU):
         super().__init__()

         self.n_inp = n_inp
         self.n_out = n_out
         self.n_neurons = [n_inp] + n_hidden_neurons + [n_out]
         self.hidden_layers = len(n_hidden_neurons)
         self.layers = torch.nn.ModuleList()
         self.activation = activation_fn

         # construct the network
         # -2 because we add the last layer manually to avoid an activatin there
         for i in range(len(self.n_neurons)-2):
             self.layers.append(torch.nn.Linear(self.n_neurons[i], self.n_neurons[i+1]))
             self.layers.append(self.activation())
         self.layers.append(nn.Linear(self.n_neurons[-2], self.n_neurons[-1]))

    def forward(self, x):
         # loop through all layer but last
         for layer in self.layers:
             x = layer(x)
         return x
