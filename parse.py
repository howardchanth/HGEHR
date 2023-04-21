from torch import optim, nn
import torch.nn.functional as F

from models import (
    GCN,
    GAT,
    GIN,
    HAN,
    HGT,
    HeteroRGCN,
    BGCN
)


def parse_optimizer(config_optim, model):
    opt_method = config_optim["opt_method"].lower()
    alpha = config_optim["lr"]
    weight_decay = config_optim["weight_decay"]
    if opt_method == "adagrad":
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=alpha,
            lr_decay=weight_decay,
            weight_decay=weight_decay,
        )
    elif opt_method == "adadelta":
        optimizer = optim.Adadelta(
            model.parameters(),
            lr=alpha,
            weight_decay=weight_decay,
        )
    elif opt_method == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=alpha,
            weight_decay=weight_decay,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=alpha,
            weight_decay=weight_decay,
        )
    return optimizer


def parse_gnn_model(config_gnn, g, tasks=None):
    gnn_name = config_gnn["name"]

    if gnn_name == "GAT":
        n_layers = config_gnn["num_layers"]
        n_heads = config_gnn["num_heads"]
        n_out_heads = config_gnn["num_out_heads"]
        heads = ([n_heads] * n_layers) + [n_out_heads]
        return GAT(
            n_layers=config_gnn["num_layers"],
            in_dim=config_gnn["in_dim"],
            hidden_dim=config_gnn["hidden_dim"],
            out_dim=config_gnn["out_dim"],
            heads=heads,
            activation=F.leaky_relu,
            feat_drop=config_gnn["feat_drop"],
            attn_drop=config_gnn["attn_drop"],
            negative_slope=config_gnn["negative_slope"],
            residual=False,
            graph_pooling_type=config_gnn["graph_pooling_type"]
        )
    elif gnn_name == "GCN":
        return GCN(
            in_dim=config_gnn["in_dim"],
            hidden_dim=config_gnn["hidden_dim"],
            out_dim=config_gnn["out_dim"],
            n_layers=config_gnn["num_layers"],
            activation=F.relu,
            dropout=config_gnn["feat_drop"],
            graph_pooling_type=config_gnn["graph_pooling_type"]
        )
    elif gnn_name == "GIN":
        return GIN(
            input_dim=config_gnn["in_dim"],
            hidden_dim=config_gnn["hidden_dim"],
            out_dim=config_gnn["out_dim"],
            num_layers=config_gnn["num_layers"],
            num_mlp_layers=config_gnn["num_mlp_layers"],
            final_dropout=config_gnn["feat_drop"],
            graph_pooling_type=config_gnn["graph_pooling_type"],
            neighbor_pooling_type=config_gnn["neighbor_pooling_type"]
        )
    elif gnn_name == "HAN":
        n_layers = config_gnn["num_layers"]
        n_heads = config_gnn["num_heads"]
        n_out_heads = config_gnn["num_out_heads"]
        heads = ([n_heads] * n_layers) + [n_out_heads]
        return HAN(
            num_meta_paths=config_gnn["num_meta_paths"],
            in_size=config_gnn["in_dim"],
            hidden_size=config_gnn["hidden_dim"],
            out_size=config_gnn["out_dim"],
            num_heads=heads,
            dropout=config_gnn["feat_drop"]
        )
    elif gnn_name == "HGT":
        return HGT(
            g,
            n_inp=config_gnn["in_dim"],
            n_hid=config_gnn["hidden_dim"],
            n_out=config_gnn["out_dim"],
            n_layers=config_gnn["num_layers"],
            n_heads=config_gnn["num_heads"],
            tasks=tasks
        )
    elif gnn_name == "HetRGCN":
        n_node_types = config_gnn["n_node_types"]
        etypes = config_gnn["edge_types"]
        canonical_etypes = [
            (str(s), r, str(t))
            for r in etypes
            for s in range(n_node_types)
            for t in range(n_node_types)
        ]
        node_dict = {str(i): i for i in range(n_node_types)}
        canonical_etypes = {et: str(i) for i, et in enumerate(canonical_etypes)}
        return HeteroRGCN(
            in_dim=config_gnn["in_dim"],
            hidden_dim=config_gnn["hidden_dim"],
            out_dim=config_gnn["out_dim"],
            n_layers=config_gnn["num_layers"],
            etypes=canonical_etypes,
            node_dict=node_dict,
            graph_pooling_type=config_gnn["graph_pooling_type"],
        )
    elif gnn_name == "BGCN":

        priors = {
            'prior_mu': config_gnn["prior_mu"],
            'prior_sigma': config_gnn["prior_sigma"],
            'posterior_mu_initial': config_gnn["posterior_mu_initial"],
            'posterior_rho_initial': config_gnn["posterior_rho_initial"],
        }

        return BGCN(
            in_dim=config_gnn["in_dim"],
            hidden_dim=config_gnn["hidden_dim"],
            out_dim=config_gnn["out_dim"],
            n_layers=config_gnn["num_layers"],
            activation=F.relu,
            dropout=config_gnn["feat_drop"],
            priors=priors
        )
    else:
        raise NotImplementedError("This GNN model is not implemented")


def parse_loss(config_train):
    loss_name = config_train["loss"]

    if loss_name == "BCE":
        return nn.BCELoss()
    elif loss_name == "CE":
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("This Loss is not implemented")