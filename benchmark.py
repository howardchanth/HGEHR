import yaml
from utils import ordered_yaml

import pickle
import random
import torch

from trainers import (
    GNNTrainer,
    CausalGNNTrainer,
    BaselinesTrainer
)


def benchmark_tasks(config):
    # Initialize baseline models
    with open(config["datasets"]["dataset_path"], 'rb') as inp:
        unp = pickle.Unpickler(inp)
        mimic3base = unp.load()

    for method in [
        # "DrAgent",
        # "StageNet",
        # "SparcNet",
        # "AdaCare",
        # "Transformer",
        # "RNN",
        # "ConCare",
        "GRSAP",
    ]:
        for task in [
            # "readm",
            # "mort_pred",
            # "los",
            "drug_rec"
        ]:
            config["train"]["baseline_name"] = method
            config["train"]["task"] = task
            config["checkpoint"]["path"] = f"./checkpoints/{method}/{task}/"
            print(f"Training {method} on task {task}")

            trainer = BaselinesTrainer(config, mimic3base)
            trainer.train()

# Set seed
seed = 611
random.seed(seed)
torch.manual_seed(seed)

config_file = "Baselines_MIMIC3.yml"
config_path = f"./configs/{config_file}"

with open(config_path, mode='r') as f:
    loader, _ = ordered_yaml()
    config = yaml.load(f, loader)
    print(f"Loaded configs from {config_path}")

if __name__ == "__main__":
    benchmark_tasks(config)
