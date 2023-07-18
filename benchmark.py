import yaml
from utils import ordered_yaml

import pickle
import random
import torch
import wandb

from trainers import (
    GNNTrainer,
    CausalGNNTrainer,
    BaselinesTrainer
)


def benchmark_baselines(config):
    # Initialize baseline models
    with open(config["datasets"]["dataset_path"], 'rb') as inp:
        unp = pickle.Unpickler(inp)
        mimic3base = unp.load()

    for method in [
        "DrAgent",
        "StageNet",
        "AdaCare",
        # "Transformer",
        "RNN",
        "ConCare",
        "GRSAP",
        "Deepr",
        "MICRON",
        "GAMENet",
        "MoleRec",
        "SafeDrug",
        # "SparcNet",
    ]:
        for task in [
            # "readm",
            # "mort_pred",
            # "los",
            "drug_rec"
        ]:
            config["train"]["baseline_name"] = method
            config["train"]["task"] = task
            dataset_name = config["datasets"]["name"]
            config["checkpoint"]["path"] = f"./checkpoints/{method}/{dataset_name}/{task}/"
            print(f"Training {method} on task {task}")

            trainer = BaselinesTrainer(config, mimic3base)
            trainer.train()
            del trainer


def benchmark_gnns(config):
    # Load GNN configs
    with open("./configs/GNN/GNN_MIMIC4_Configs.yml", mode='r') as f:
        loader, _ = ordered_yaml()
        gnn_config = yaml.load(f, loader)

    for archi in [
        # "GCN",
        # "GAT",
        "GIN",
        "HetRGCN",
        # "HGT"
    ]:
        config["GNN"] = gnn_config[archi]
        dataset_name = config["datasets"]["name"]
        config["name"] = f"{archi}_MTCausal_MIMIC{dataset_name[-1]}_RMDL"
        config["checkpoint"]["path"] = f"./checkpoints/GNN_ablation/{dataset_name}/{archi}/"
        config["logging"]["tags"] += [archi]

        trainer = CausalGNNTrainer(config)
        trainer.train()
        wandb.finish()
        del trainer


def benchmark_dropouts(config):
    for dp in [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6
    ]:
        config["GNN"]["feat_drop"] = dp
        config["name"] = f"HGT_MTCausal_MIMIC3_RMDL_dp{dp}"
        dataset_name = config["datasets"]["name"]
        config["checkpoint"]["path"] = f"./checkpoints/Dropout_ablation/{dataset_name}/{dp}/"
        config["logging"]["tags"] += ["abl_dropout"]

        trainer = CausalGNNTrainer(config)
        trainer.train()
        wandb.finish()
        del trainer


def benchmark_hidden_dim(config):
    for dim in [
        16, 32, 64, 128, 256
    ]:
        config["GNN"]["hidden_dim"] = dim
        config["name"] = f"HGT_MTCausal_MIMIC3_RMDL_dim{dim}"
        dataset_name = config["datasets"]["name"]
        config["checkpoint"]["path"] = f"./checkpoints/Hidden_Dim_ablation/{dataset_name}/{dim}/"
        config["logging"]["tags"] += ["abl_dim"]

        trainer = CausalGNNTrainer(config)
        trainer.train()
        wandb.finish()
        del trainer


def benchmark_reg(config):
    for reg in [
        0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.1, 1
    ]:
        config["train"]["reg"] = reg
        config["name"] = f"HGT_MTCausal_MIMIC3_RMDL_reg{reg}"
        dataset_name = config["datasets"]["name"]
        config["checkpoint"]["path"] = f"./checkpoints/Reg_Coeff_ablation/{dataset_name}/{reg}/"
        config["logging"]["tags"] += ["abl_reg"]

        trainer = CausalGNNTrainer(config)
        trainer.train()
        wandb.finish()
        del trainer

# Set seed
seed = 611
random.seed(seed)
torch.manual_seed(seed)

config_file = "HGT_Causal_MIMIC3.yml"
config_path = f"./configs/{config_file}"

with open(config_path, mode='r') as f:
    loader, _ = ordered_yaml()
    config = yaml.load(f, loader)
    print(f"Loaded configs from {config_path}")

if __name__ == "__main__":
    # benchmark_baselines(config)[
    # benchmark_gnns(config)
    # benchmark_dropouts(config)
    # benchmark_hidden_dim(config)
    benchmark_reg(config)