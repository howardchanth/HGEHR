"""
Trainer of baseline models
"""

import pickle

import pyhealth
from pyhealth.trainer import Trainer
from pyhealth.tasks import (
    drug_recommendation_mimic3_fn,
    readmission_prediction_mimic3_fn,
    mortality_prediction_mimic3_fn,
    length_of_stay_prediction_mimic3_fn,
    drug_recommendation_mimic4_fn,
    readmission_prediction_mimic4_fn,
    mortality_prediction_mimic4_fn,
    length_of_stay_prediction_mimic4_fn
)
from pyhealth.datasets import split_by_patient, get_dataloader, split_by_visit

from collections import OrderedDict

from .trainer import Trainer as MyTrainer
from parse import parse_baselines

import plotly.graph_objects as go


class BaselinesTrainer(MyTrainer):
    def __init__(self, config: OrderedDict, mimic3base):
        super().__init__(config)

        # Load graph and task labels
        dataset_path = self.config_data["dataset_path"]
        baseline_name = self.config_train["baseline_name"]
        task = self.config_train["task"]
        metrics = self.set_mode_metrics(task)

        mimic3sample = self.set_task(task, mimic3base)  # use default task
        train_ds, val_ds, test_ds = split_by_visit(mimic3sample, [0.9, 0.1, 0])

        # create dataloaders (torch.data.DataLoader)
        self.train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
        self.val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
        self.test_loader = get_dataloader(test_ds, batch_size=512, shuffle=False)

        model = parse_baselines(mimic3sample, baseline_name, self.mode, self.label_key)
        self.trainer = Trainer(
            model=model,
            metrics=metrics,
            output_path=self.checkpoint_manager.path
        )

    def train(self):

        self.trainer.train(
            train_dataloader=self.train_loader,
            val_dataloader=self.val_loader,
            epochs=self.n_epoch,
            monitor=self.monitor,
        )

    def visualize_embeddings(self):

        from sklearn.manifold import Isomap, TSNE

        layout = go.Layout(
            autosize=False,
            width=600,
            height=600)
        fig = go.Figure(layout=layout)

        data_batch = next(iter(self.test_loader))
        embeddings = self.trainer.model.embeddings

        # TODO: Get patient embedding from one iteration
        offset = 0
        for k, v in self.node_dict.items():
            indices = [i for i in range(offset, offset + 250)]
            tsne = TSNE(n_components=2)
            embeddings_2d = tsne.fit_transform(embeddings[indices])
            offset += len(v)

            fig.add_trace(go.Scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], mode='markers', name=k))

        fig.write_image()


    def set_task(self, task, base_dataset):
        name = self.config_data["name"]
        if task == "readm":
            sample_dataset = base_dataset.set_task(task_fn=globals()[f"readmission_prediction_{name}_fn"])
        elif task == "mort_pred":
            sample_dataset = base_dataset.set_task(task_fn=globals()[f"mortality_prediction_{name}_fn"])
        elif task == "los":
            sample_dataset = base_dataset.set_task(task_fn=globals()[f"length_of_stay_prediction_{name}_fn"])
        elif task == "drug_rec":
            sample_dataset = base_dataset.set_task(task_fn=globals()[f"drug_recommendation_{name}_fn"])
        else:
            raise NotImplementedError

        return sample_dataset

    def set_mode_metrics(self, task):
        if task in ["readm", "mort_pred"]:
            self.mode = "binary"
            self.monitor = "roc_auc"
            self.label_key = "label"
            return ["accuracy", "pr_auc", "roc_auc", "f1"]
        elif task == "los":
            self.mode = "multiclass"
            self.monitor = "accuracy"
            self.label_key = "label"
            return ["accuracy", "f1_macro", "roc_auc_weighted_ovo"]
        elif task == "drug_rec":
            self.mode = "multilabel"
            self.monitor = "pr_auc_weighted"
            self.label_key = "drugs"
            return ["accuracy", "f1_macro", "roc_auc_samples", "jaccard_weighted", "pr_auc_weighted"]
