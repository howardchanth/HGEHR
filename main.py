import yaml
from utils import ordered_yaml, load_config

import random
import torch

from trainers import (
    GNNTrainer,
    CausalGNNTrainer,
    BaselinesTrainer
)

from pretrainers import (
    Pretrainer
)

# Set seed
seed = 612
random.seed(seed)
torch.manual_seed(seed)

#############################################################
# Set modes:
# train: initialize trainer for classification
# pretrain: pretrain the node embeddings
# eval: Evaluate the trained model
#############################################################
mode = "train"
# mode = "pretrain"


def main():
    if mode == "train":
        config_name = "HGT_Causal_MIMIC4.yml"
        config = load_config(config_name)

        if config["train_type"] == "gnn":
            trainer = GNNTrainer(config)
        elif config["train_type"] == "causal-gnn":
            trainer = CausalGNNTrainer(config)
        elif config["train_type"] == "baseline":
            trainer = BaselinesTrainer(config)
        else:
            raise NotImplementedError("This type of model is not implemented")
        trainer.train()
    elif mode == "pretrain":
        config_name = "MIMIC4_TransE.yml"
        config = load_config(config_name, "./configs/pretrain/")

        pretrainer = Pretrainer(config)
        pretrainer.train()


if __name__ == "__main__":
    main()


