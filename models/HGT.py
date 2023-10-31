import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .GNN import GNN


class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads, dropout=0.2, use_norm=False):
        super(HGTLayer, self).__init__()

        num_types = len(ntypes)
        num_relations = len(etypes)

        self.e_dict = {s: i for (i, s) in enumerate(etypes)}

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def edge_attention(self, edges):
        etype = self.e_dict[edges.canonical_etype[1]]
        relation_att = self.relation_att[etype]
        relation_pri = self.relation_pri[etype]
        relation_msg = self.relation_msg[etype]
        key = torch.bmm(edges.src['k'].transpose(1, 0), relation_att).transpose(1, 0)
        att = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
        val = torch.bmm(edges.src['v'].transpose(1, 0), relation_msg).transpose(1, 0)
        return {'a': att, 'v': val}

    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}

    def forward(self, G, inp_key, out_key):
        node_dict = {tp: i for i, tp in enumerate(G.ntypes)}
        edge_dict = {tp: i for i, tp in enumerate(G.etypes)}
        for srctype, etype, dsttype in G.canonical_etypes:
            k_linear = self.k_linears[node_dict[srctype]]
            v_linear = self.v_linears[node_dict[srctype]]
            q_linear = self.q_linears[node_dict[dsttype]]

            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)

            G.apply_edges(func=self.edge_attention, etype=etype)
        G.multi_update_all({etype: (self.message_func, self.reduce_func) \
                            for etype in edge_dict}, cross_reducer='mean')
        for ntype in G.ntypes:
            n_id = node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            trans_out = self.a_linears[n_id](G.nodes[ntype].data['t'])
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1 - alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[n_id](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class HGT(GNN):
    def __init__(self, G, n_inp, n_hid, n_out, n_layers, n_heads, tasks, causal, dropout, use_norm=True):

        self.n_heads = n_heads
        self.use_norm = use_norm
        self.ntypes = G.ntypes
        self.etypes = G.etypes

        super(HGT, self).__init__(n_inp, n_hid, n_out, n_layers, F.relu, dropout, tasks, causal)

        self.adapt_ws = nn.ModuleList()
        for t in range(len(G.ntypes)):
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))

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
        for i in range(self.n_layers):
            layers.append(HGTLayer(self.hidden_dim, self.hidden_dim, self.ntypes, self.etypes, self.n_heads, self.dor, use_norm=self.use_norm))

        return layers

    def get_logit(self, g, h=None, causal=False):
        layers = self.layers if not causal else self.rand_layers
        for n_id, ntype in enumerate(self.ntypes):
            g.nodes[ntype].data['h'] = F.gelu(self.adapt_ws[n_id](g.nodes[ntype].data['feat']))
        for i in range(self.n_layers):
            layers[i](g, 'h', 'h')
        return g.ndata['h']
