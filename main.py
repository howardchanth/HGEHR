import yaml
from utils import ordered_yaml

import argparse

import random
import torch

from trainers import (
    GNNTrainer
)

# Set seed
seed = 612
random.seed(seed)
torch.manual_seed(seed)

#############################################################
# Set modes:
# train: initialize trainer for classification
# eval: Evaluate the trained model quantitatively
# construct_graph: Construct graphs from WSI and save to disk
# graph_explain: Explain the GNN and plot the results
#############################################################
mode = "train"


def main():
    config_file = "GCN_MIMIC3.yml"
    config_path = f"./configs/{config_file}"

    with open(config_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {config_path}")

    if mode == "train":
        if config["train_type"] == "gnn":
            trainer = GNNTrainer(config)
        else:
            raise NotImplementedError("This type of model is not implemented")
        trainer.train()


if __name__ == "__main__":
    main()


