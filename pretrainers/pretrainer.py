import dgl
import torch
from torch import nn
from dgl.nn import TransE
from dgl.sampling import global_uniform_negative_sampling

import pickle
from tqdm import tqdm

from data import load_graph


class Pretrainer:
    def __init__(self, config):
        graph_path = config["graph_path"]
        label_path = config["labels_path"]
        self.graph, _, _, _ = load_graph(graph_path, label_path)

        sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
        sampler = dgl.dataloading.as_edge_prediction_sampler(
            sampler,
            negative_sampler=dgl.dataloading.negative_sampler.Uniform(5)
        )
        edge_dict = {k: torch.arange(len(self.graph.edges(etype=k)[0])) for k in self.graph.etypes}
        self.dataloader = dgl.dataloading.DataLoader(
            self.graph, edge_dict, sampler,
            batch_size=1024, shuffle=True, drop_last=False, num_workers=0)

        self.feat = nn.ParameterDict()
        self.feat.update({k: nn.Parameter(v) for k, v in self.graph.ndata['feat'].items()})
        self.optimizer = torch.optim.Adam(list(self.feat.values()), lr=0.05)

        self.output_path = config["graph_output_path"]
        self.margin = config["margin"]

        self.scorer = TransE(num_rels=4, feats=128)

        self.n_epoch = config["n_epoch"]

    def compute_scores(self, pos_g, neg_g):
        pos_scores = []
        neg_scores = []
        for i, (src_type, rel, dest_type) in enumerate(pos_g.canonical_etypes):
            src, dest = pos_g.edges(etype=rel)
            hs = self.graph.srcdata['feat'][src_type][src]
            hd = self.graph.dstdata['feat'][dest_type][dest]
            s = self.scorer(hs, hd, (torch.ones_like(hs[:, 0]) * i).to(torch.int32))
            pos_scores.append(s)

            src, dest = neg_g.edges(etype=rel)
            hs = self.graph.srcdata['feat'][src_type][src]
            hd = self.graph.dstdata['feat'][dest_type][dest]
            s = self.scorer(hs, hd, (torch.ones_like(hs[:, 0]) * i).to(torch.int32))
            neg_scores.append(s)

        pos_scores = torch.cat(pos_scores)
        neg_scores = torch.cat(neg_scores)

        return pos_scores, neg_scores

    def train(self):
        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            epoch_range = tqdm(self.dataloader)
            res = 0
            for input_nodes, pos_g, neg_g, blocks in epoch_range:
                pos_score, neg_score = self.compute_scores(pos_g, neg_g)
                pos_score = pos_score.tile(5)
                loss = (pos_score - neg_score + self.margin).relu().sum()

                self.optimizer.zero_grad()
                loss.backward()
                res += loss.item()
                self.optimizer.step()
                self.graph.ndata['feat'] = {k: v.detach().cpu() for k, v in self.feat.items()}

                epoch_range.set_description_str(
                    "Epoch {} | loss: {:.4f}".format(
                        epoch, loss.item(),
                    )
                )

            training_range.set_description_str(
                "Epoch {} | loss: {:.4f}".format(
                    epoch, res),
            )

        # Save embeddings
        feat = {k: v.detach().cpu() for k, v in self.feat.items()}
        self.graph.ndata['feat'] = feat

        self.save_graph()

    def save_graph(self):
        with open(self.output_path, 'wb') as outp:
            pickle.dump(self.graph, outp, pickle.HIGHEST_PROTOCOL)