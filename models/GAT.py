"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn

import dgl
from dgl.nn import GATConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling

from .GNN import GNN

class GAT(GNN):
    def __init__(self,
                 n_layers,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 tasks,
                 causal):

        self.heads = heads
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual

        super().__init__(in_dim, hidden_dim, out_dim, n_layers, activation, feat_drop, tasks, causal)

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

        self.set_embeddings(h)

        return out

    def get_logit(self, g, h, causal=False):
        layers = self.layers if not causal else self.rand_layers
        for i, layer in enumerate(layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            if i != len(layers) - 1:
                h = h.flatten(1)
            else:
                h = h.mean(1)

        self.embeddings = h

        return h

    def get_layers(self):

        layers = nn.ModuleList()
        for l in range(self.n_layers):
            if l == 0:
                # input projection (no residual)
                layers.append(GATConv(
                    self.in_dim, self.hidden_dim, self.heads[0],
                    self.dor, self.attn_drop, self.negative_slope, False, self.activation))
            else:
                # due to multi-head, the in_dim = num_hidden * num_heads
                layers.append(GATConv(
                    self.hidden_dim * self.heads[l-1], self.hidden_dim, self.heads[l],
                    self.dor, self.attn_drop, self.negative_slope, self.residual, self.activation))

        return layers