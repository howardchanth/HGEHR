import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from .GNN import GNN


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etype_dict):
        super(HeteroRGCNLayer, self).__init__()
        self.etype_dict = etype_dict
        # W_r for each relation
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etype_dict.values()
        })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        new_feat_dict = {k: [] for k in feat_dict.keys()}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Get etype
            local_etype = self.etype_dict[(srctype, etype, dsttype)]
            # Compute W_r * h
            Wh = self.weight[local_etype](feat_dict[srctype])

            new_feat_dict[srctype].append(Wh)

        for tp, tensors in new_feat_dict.items():
            new_feat_dict[tp] = feat_dict[tp]

        # return the updated node feature dictionary
        return new_feat_dict


class HeteroRGCN(GNN):
    def __init__(self, G, in_dim, hidden_dim, out_dim, n_layers, tasks, causal):

        self.ntypes = G.ntypes
        self.etypes = G.etypes
        self.canonical_etype = G.canonical_etypes
        self.etype_dict = {k: str(i) for i, k in enumerate(self.canonical_etype)}

        super(HeteroRGCN, self).__init__(in_dim, hidden_dim, out_dim, n_layers, F.relu, 0.2, tasks, causal)

        self.adapt_ws = nn.ModuleList()
        for t in range(len(G.ntypes)):
            self.adapt_ws.append(nn.Linear(in_dim, hidden_dim))

    def forward(self, g, out_key, task):

        logits = self.get_logit(g)
        self.embeddings = torch.cat(list(logits.values()))
        out = self.out[task](logits[out_key])

        if self.causal:
            feat_rand = self.get_logit(g, causal=True)
            feat_interv = {k: logits[k] + feat_rand[k] for k in feat_rand.keys()}
            out_interv = self.out[task](feat_interv[out_key])
            feat_rand = torch.cat(list(feat_rand.values()))
            return out, feat_rand, out_interv


        return out

    def get_layers(self):
        layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layers.append(HeteroRGCNLayer(self.hidden_dim, self.hidden_dim, self.etype_dict))
        return layers

    def get_logit(self, g, h=None, causal=False):
        layers = self.layers if not causal else self.rand_layers
        for n_id, ntype in enumerate(self.ntypes):
            g.nodes[ntype].data['h'] = torch.tanh(self.adapt_ws[n_id](g.nodes[ntype].data['feat']))
        for i in range(self.n_layers):
            feat_dict = g.ndata['h']
            g.ndata['h'] = layers[i](g, feat_dict)
        return g.ndata['h']

