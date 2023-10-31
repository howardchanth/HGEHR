import math
from abc import ABC
from collections import OrderedDict

import wandb

from checkpoint import CheckpointManager


class Trainer(ABC):
    def __init__(self, config: OrderedDict) -> None:
        # Categorize configurations
        self.config = config
        self.config_data = config["datasets"]
        self.config_train = config['train']
        self.config_optim = config['optimizer']
        self.config_checkpoint = config['checkpoint']

        # Read name from configs
        self.name = config['name']
        self.gnn = None

        # Define checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.config_checkpoint['path'])
        self.save_steps = self.config_checkpoint["save_checkpoint_freq"]

        # Training Settings
        self.n_epoch = self.config_train['num_epochs']
        self.batch_size = self.config_train['batch_size']

        # Load device for training
        self.gpu_ids = config['gpu_ids']
        self.device = "cuda" if config['gpu_ids'] else "cpu"
        self.use_gpu = True if self.device == "cuda" else False

        self.init_temperature = 1

    def train(self) -> None:
        raise NotImplementedError

    def initialize_logger(self, name, notes=""):
        tags = self.config["logging"]["tags"]
        mode = self.config["logging"]["mode"]
        wandb.init(name=name,
                   project='MT_EHR',
                   notes=notes,
                   mode=mode,
                   config=self.config,
                   tags=tags)

    def anneal_temperature(self, epoch):
        self.temperature = max(0.1, math.exp(- epoch * self.init_temperature))
