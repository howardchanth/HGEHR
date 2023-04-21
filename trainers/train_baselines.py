"""
Trainer of baseline models
"""

import pyhealth
from pyhealth.trainer import Trainer

from collections import OrderedDict

from .trainer import Trainer as MyTrainer

from data import load_graph

class BaselinesTrainer(MyTrainer):
    def __init__(self, config: OrderedDict):
        super().__init__(config)

        # os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_ids']

        # Load graph and task labels
        graph_path = self.config_data["graph_path"]
        dataset_path = self.config_data["dataset_path"]
        entity_mapping = self.config_data["entity_mapping"]
        self.graph = load_graph(graph_path, dataset_path, entity_mapping)

        # Initialize baseline models


    def train(self):