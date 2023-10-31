import os
from collections import OrderedDict

import dgl
from tqdm import tqdm

import torch
from torch.nn import functional as F

import numpy as np

from .trainer import Trainer
from parse import (
    parse_optimizer,
    parse_gnn_model,
    parse_loss
)

from data import load_graph
from utils import metrics


class GNNTrainer(Trainer):
    def __init__(self, config: OrderedDict):
        super().__init__(config)

        self.config_gnn = config["GNN"]

        # Initialize GNN model and optimizer
        self.tasks = ["readm"]

        # Load graph, labels and splits
        graph_path = self.config_data["graph_path"]
        dataset_path = self.config_data["dataset_path"]
        labels_path = self.config_data["labels_path"]
        entity_mapping = self.config_data["entity_mapping"]
        self.graph, self.labels, self.train_mask, self.test_mask = load_graph(graph_path, labels_path)

        # Transform the graph
        self.graph = dgl.AddReverse()(self.graph)

        # Read node_dict
        self.node_dict = {}
        for tp in self.graph.ntypes:
            self.node_dict.update({tp: torch.arange(self.graph.num_nodes(tp))})

        self.gnn = parse_gnn_model(self.config_gnn, self.graph, self.tasks).to(self.device)
        self.optimizer = parse_optimizer(self.config_optim, self.gnn)

    def train(self) -> None:
        print(f"Start training GNN")

        training_range = tqdm(range(self.n_epoch), nrows=3)

        for epoch in training_range:
            self.gnn.train()
            epoch_stats = {"Epoch": epoch + 1}
            preds, labels = None, None

            # Perform aggregation on visits
            self.optimizer.zero_grad()
            d = self.node_dict.copy()
            for t in self.tasks:
                all_preds = []
                # for indx in range(0, n_visits, self.batch_size):
                    # high = min(indx + self.batch_size, n_visits - 1)
                indices = self.train_mask[t]
                d["visit"] = self.node_dict["visit"][indices]
                sg = self.graph.subgraph(d).to(self.device)
                preds = self.gnn(sg, "visit", t)
                labels = torch.LongTensor([self.labels[t][i] for i in indices]).to(self.device)
                loss = F.cross_entropy(preds, labels)

            loss.backward()
            self.optimizer.step()

            train_metrics = metrics(preds, labels, average="binary")

            # Perform validation and testing
            self.checkpoint_manager.save_model(self.gnn.state_dict())
            test_metrics = self.evaluate()

            training_range.set_description_str("Epoch {} | loss: {:.4f}| Train AUC: {:.4f} | Test AUC: {:.4f} | Test ACC: {:.4f} ".format(
                epoch, loss.item(), train_metrics["train_auroc"], test_metrics["test_auroc"], test_metrics["test_accuracy"]))

            epoch_stats.update({"Train Loss: ": loss.item()})
            epoch_stats.update(train_metrics)
            epoch_stats.update(test_metrics)

            # State dict of the model including embeddings
            self.checkpoint_manager.write_new_version(
                self.config,
                self.gnn.state_dict(),
                epoch_stats
            )

            # Remove previous checkpoint
            self.checkpoint_manager.remove_old_version()

    def evaluate(self):
        self.gnn.eval()
        for t in self.tasks:
            indices = self.test_mask[t]
            labels = torch.LongTensor([self.labels[t][i] for i in indices]).to(self.device)

            d = self.node_dict.copy()
            d["visit"] = self.node_dict["visit"][indices]
            sg = self.graph.subgraph(d).to(self.device)
            with torch.no_grad():
                preds = self.gnn(sg, "visit", t)

        test_metrics = metrics(preds, labels, average="binary", prefix="test")

        return test_metrics

    def get_masks(self, g: dgl.DGLGraph, train: bool, task: str):
        if train:
            masks = self.train_mask[task]
            labels = [self.labels[task][v] for v in masks]
        else:
            masks = self.test_mask[task]
            labels = [self.labels[task][v] for v in masks]

        m = {}

        for tp in g.ntypes:
            if tp == "visit":
                m[tp] = torch.from_numpy(masks.astype("int32"))
            else:
                m[tp] = torch.zeros(0)

        return m

    def get_labels(self, train: bool, task: str):
        if train:
            masks = self.train_mask[task]
            labels = [self.labels[task][v] for v in masks]
        else:
            masks = self.test_mask[task]
            labels = [self.labels[task][v] for v in masks]

        return masks, labels

    def up_sample(self, scores, label):
        """
        Up sample labels to ensure data balance
        :param scores:
        :param label:
        :return:
        """

    # def train_one_step(self, label):
    #     self.optimizer.zero_grad()
    #
    #     for t in self.tasks:
    #         pred = self.gnn(self.graph, "visit", t)
    #         prob = F.softmax(pred)
    #
    #     loss = F.cross_entropy(pred, label)
    #
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     accuracy = acc(pred, label)
    #
    #     pred = pred.detach().cpu().numpy().argmax(axis=1)
    #     prob = prob.detach().cpu().numpy()
    #     label = label.detach().cpu().numpy()
    #
    #     return loss.item(), accuracy, pred, prob, label
