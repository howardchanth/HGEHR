"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

from .GNN import GNN

class GCN(GNN):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 tasks,
                 causal=False):
        super().__init__(in_dim, hidden_dim, out_dim, n_layers, activation, dropout, tasks, causal)

        self.n_layers = n_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

    def forward(self, g: dgl.DGLHeteroGraph, nt, task):
        g = dgl.to_homogeneous(g, ndata=["feat"], store_type=True)
        g = dgl.add_self_loop(g)

        h = g.ndata["feat"]
        logits = self.get_logit(g, h)
        h = self.out[task](logits)
        out = h[g.ndata["_TYPE"] == 4]

        if self.causal:
            h = g.ndata["feat"]
            feat_rand = self.get_logit(g, h, True)
            feat_interv = logits + feat_rand
            out_interv = self.out[task](feat_interv)
            return out, feat_rand, out_interv

        return out

    def get_logit(self, g, h, causal=False):
        layers = self.layers if not causal else self.rand_layers
        for i, layer in enumerate(layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        self.set_embeddings(h)

        return h

    def get_layers(self):
        layers = nn.ModuleList()
        layers.append(GraphConv(self.in_dim, self.hidden_dim, activation=self.activation))
        # hidden layers
        for i in range(self.n_layers - 1):
            layers.append(GraphConv(self.hidden_dim, self.hidden_dim, activation=self.activation))

        return layers
