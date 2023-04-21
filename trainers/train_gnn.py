import os
from collections import OrderedDict

import dgl
from tqdm import tqdm

import torch
from torch.nn import functional as F

from dgl.dataloading import DataLoader
from dgl.dataloading import MultiLayerFullNeighborSampler

import numpy as np

from .trainer import Trainer
from parse import (
    parse_optimizer,
    parse_gnn_model,
    parse_loss
)

from data import load_graph
from utils import acc, metrics


class GNNTrainer(Trainer):
    def __init__(self, config: OrderedDict):
        super().__init__(config)

        # Initialize GNN model and optimizer
        self.tasks = ["mort_pred"]

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

    def train_one_step(self):
        self.optimizer.zero_grad()

        for t in self.tasks:
            pred = self.gnn(self.graph, "visit", t)
            prob = F.softmax(pred)

        loss = F.cross_entropy(pred, label)

        loss.backward()
        self.optimizer.step()

        accuracy = acc(pred, label)

        pred = pred.detach().cpu().numpy().argmax(axis=1)
        prob = prob.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        return loss.item(), accuracy, pred, prob, label

    def evaluate(self):

        for t in self.tasks:
            indices = self.test_mask[t]
            labels = torch.LongTensor([self.labels[t][i] for i in indices]).to(self.device)

            d = self.node_dict.copy()
            d["visit"] = self.node_dict["visit"][indices]
            sg = self.graph.subgraph(d).to(self.device)
            with torch.no_grad():
                preds = self.gnn(sg, "visit", t)

        accuracy = acc(preds, labels)
        probs = preds.softmax(1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        precision, recall, f1_score, auc = metrics(probs, labels, average="binary")

        return accuracy, f1_score, precision, recall, auc

    def train(self) -> None:
        print(f"Start training Homogeneous GNN")

        training_range = tqdm(range(self.n_epoch), nrows=3)
        metrics_log = tqdm(total=0, position=1, bar_format='{desc}')

        for epoch in training_range:
            self.gnn.train()

            res = 0
            pred_list = []
            prob_list = []
            label_list = []
            accuracy_list = []

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

            accuracy = acc(preds, labels)
            probs = preds.softmax(1).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            precision, recall, f1_score, train_auc = metrics(probs, labels, average="binary")

            # Perform validation and testing
            self.checkpoint_manager.save_model(self.gnn.state_dict())
            test_acc, test_f1, test_prec, test_recall, test_auc = self.evaluate()

            training_range.set_description_str("Epoch {} | loss: {:.4f}".format(epoch, loss.item()))
            metrics_list = (accuracy, f1_score, precision, recall, train_auc,
                            test_acc, test_f1, test_prec, test_recall, test_auc)
            metrics_log.set_description_str(
                "Metrics ==> [Acc: {:.4f} | F1: {:.4f} | Ps: {:.4f} | Rec: {:.4f} | AUC: {:.4f} |"
                " Test Acc: {:.4f} | Test F1: {:.4f} | Test Ps: {:.4f} | Test Rec: {:.4f} | Test AUC: {:.4f}]".format(*metrics_list)
            )

            epoch_stats = {
                "Epoch": epoch + 1,
                "Train Loss: ": loss.item(),
                "Training Accuracy": accuracy,
                "Training Precision": precision,
                "Training Recall": recall,
                "Training F1": f1_score,
                "Training AUC": train_auc,
                "Testing Accuracy": test_acc,
                "Testing F1": test_f1,
                "Testing Precision": test_prec,
                "Testing Recall": test_recall,
                "Testing AUC": test_auc,
            }

            # State dict of the model including embeddings
            self.checkpoint_manager.write_new_version(
                self.config,
                self.gnn.state_dict(),
                epoch_stats
            )

            # Remove previous checkpoint
            self.checkpoint_manager.remove_old_version()

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