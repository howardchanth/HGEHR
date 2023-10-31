import dgl
import numpy as np
import pickle

import torch


def load_graph(graph_path, labels_path, feat_dim=128, pretrained=None):

    # Load graph from cache
    with open(graph_path, 'rb') as inp:
        unp = pickle.Unpickler(inp)
        g = unp.load()

    # Load labels
    with open(labels_path, 'rb') as inp:
        unp = pickle.Unpickler(inp)
        labels = unp.load()

    n_node_types = len(g.ntypes)

    # Set masks for entities
    if not pretrained:
        for tp in g.ntypes:
            n_nodes = g.num_nodes(tp)

            # Initialize features
            feat = torch.randn(n_nodes, feat_dim)
            g.nodes[tp].data["feat"] = feat
    else:
        with open(pretrained, 'rb') as inp:
            unp = pickle.Unpickler(inp)
            pre_g = unp.load()

            g.ndata["feat"] = pre_g.ndata["feat"].copy()
            del pre_g

    # Arrange masks by tasks
    train_masks = {}
    test_masks = {}
    for k, lb in labels.items():
        if k == "all_drugs":
            train_masks.update({k: lb})
            test_masks.update({k: lb})
            continue
        indices = np.random.permutation(len(lb))
        split = int(0.9 * len(lb))

        all_visits = np.array([k for k in lb.keys()])
        train_visits = all_visits[indices[:split]]
        test_visits = all_visits[indices[split:]]

        train_masks.update({k: train_visits})
        test_masks.update({k: test_visits})

    return g, labels, train_masks, test_masks