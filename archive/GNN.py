"""
General GNN Module
"""

import torch
from torch import nn

from dgl.nn.pytorch import GraphConv

class GNN(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()

        self.n_layers = n_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = activation

        self.layers = self.get_layers()

    def get_layers(self):
        layers = nn.ModuleList()
        layers.append(GraphConv(self.in_dim, self.hidden_dim, activation=self.activation))
        # hidden layers
        for i in range(self.n_layers - 1):
            layers.append(GraphConv(self.hidden_dim, self.hidden_dim, activation=self.activation))

        return layers

