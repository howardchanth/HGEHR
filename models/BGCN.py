import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from layers import BBBGraphConv, BBBLinear


class BGCN(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 priors,
                 graph_pooling_type="max"):
        super(BGCN, self).__init__()

        self.in_feats = in_dim
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(BBBGraphConv(in_dim, hidden_dim, activation=activation, priors=priors))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(BBBGraphConv(hidden_dim, hidden_dim, activation=activation, priors=priors))
        self.dropout = nn.Dropout(p=dropout)

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(n_layers):
            if layer == 0:
                self.linears_prediction.append(
                    BBBLinear(in_dim, out_dim))
            else:
                self.linears_prediction.append(
                    BBBLinear(hidden_dim, out_dim))

        self.drop = nn.Dropout(dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h=None):
        if h is None:
            h = g.ndata['feat']

        h_list = []
        kl = 0
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h_list.append(self.linears_prediction[i](h))
            h = layer(g, h)
            kl += layer.kl_loss()

        with g.local_scope():
            # Pool the readouts
            h = torch.stack(h_list).mean(0)
            g.ndata['h'] = h

            # Calculate graph representation by average readout.
            out = dgl.mean_nodes(g, 'h')

            return out, kl
