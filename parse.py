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

from pyhealth.models import (
    RNN,
    Transformer,
    AdaCare,
    ConCare,
    StageNet,
    Deepr,
    Agent,
    GRASP,
    SparcNet,
    MICRON,
    MoleRec,
    GAMENet,
    SafeDrug
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


def parse_gnn_model(config_gnn, g, tasks=None, causal=False):
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
            tasks=tasks,
            causal=causal,
            residual=False
        )
    elif gnn_name == "GCN":
        return GCN(
            in_dim=config_gnn["in_dim"],
            hidden_dim=config_gnn["hidden_dim"],
            out_dim=config_gnn["out_dim"],
            n_layers=config_gnn["num_layers"],
            activation=F.relu,
            dropout=config_gnn["feat_drop"],
            tasks=tasks,
            causal=causal
        )
    elif gnn_name == "GIN":
        return GIN(
            in_dim=config_gnn["in_dim"],
            hidden_dim=config_gnn["hidden_dim"],
            out_dim=config_gnn["out_dim"],
            num_layers=config_gnn["num_layers"],
            num_mlp_layers=config_gnn["num_mlp_layers"],
            final_dropout=config_gnn["feat_drop"],
            neighbor_pooling_type=config_gnn["neighbor_pooling_type"],
            tasks=tasks,
            causal=causal
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
            dropout=config_gnn["feat_drop"],
            tasks=tasks,
            causal=causal
        )
    elif gnn_name == "HetRGCN":
        return HeteroRGCN(
            g,
            in_dim=config_gnn["in_dim"],
            hidden_dim=config_gnn["hidden_dim"],
            out_dim=config_gnn["out_dim"],
            n_layers=config_gnn["num_layers"],
            tasks=tasks,
            causal=causal
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


def parse_baselines(dataset, baseline_name, mode, label_key):
    if baseline_name == "AdaCare":
        return AdaCare(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key=label_key,
            use_embedding=[True, True],
            mode=mode,
        )
    elif baseline_name == "Transformer":
        return Transformer(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key=label_key,
            mode=mode,
        )
    elif baseline_name == "ConCare":
        return ConCare(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key=label_key,
            use_embedding=[True, True],
            mode=mode
        )
    elif baseline_name == "DrAgent":
        return Agent(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key=label_key,
            mode=mode
        )
    elif baseline_name == "Deepr":
        return Deepr(
            dataset=dataset,
            feature_keys=["conditions", "procedures", "prescriptions"],
            label_key=label_key,
            mode=mode
        )
    elif baseline_name == "RNN":
        return RNN(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key=label_key,
            mode=mode,
        )
    elif baseline_name == "GRSAP":
        return GRASP(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key=label_key,
            use_embedding=[True, True],
            mode=mode
        )
    elif baseline_name == "StageNet":
        return StageNet(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key=label_key,
            mode=mode,
        )
    elif baseline_name == "SparcNet":
        return SparcNet(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key=label_key,
            mode=mode,
        )
    elif baseline_name == "MICRON":
        return MICRON(
            dataset=dataset
        )
    elif baseline_name == "MoleRec":
        return MoleRec(
            dataset=dataset
        )
    elif baseline_name == "GAMENet":
        return GAMENet(
            dataset=dataset
        )
    elif baseline_name == "SafeDrug":
        return SafeDrug(
            dataset=dataset
        )
    else:
        raise NotImplementedError("This baseline is not implemented")


def parse_loss(config_train):
    loss_name = config_train["loss"]

    if loss_name == "BCE":
        return nn.BCELoss()
    elif loss_name == "CE":
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("This Loss is not implemented")
