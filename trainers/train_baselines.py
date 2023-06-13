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
    length_of_stay_prediction_mimic3_fn
)
from pyhealth.datasets import split_by_patient, get_dataloader

from collections import OrderedDict

from .trainer import Trainer as MyTrainer
from parse import parse_baselines


class BaselinesTrainer(MyTrainer):
    def __init__(self, config: OrderedDict, mimic3base):
        super().__init__(config)

        # Load graph and task labels
        dataset_path = self.config_data["dataset_path"]
        baseline_name = self.config_train["baseline_name"]
        task = self.config_train["task"]
        metrics = self.set_mode_metrics(task)

        mimic3sample = self.set_task(task, mimic3base)  # use default task
        train_ds, val_ds, test_ds = split_by_patient(mimic3sample, [0.8, 0.1, 0.1])

        # create dataloaders (torch.data.DataLoader)
        self.train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
        self.val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
        self.test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

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

    def set_task(self, task, mimic3base):
        if task == "readm":
            mimic3sample = mimic3base.set_task(task_fn=readmission_prediction_mimic3_fn)
        elif task == "mort_pred":
            mimic3sample = mimic3base.set_task(task_fn=mortality_prediction_mimic3_fn)
        elif task == "los":
            mimic3sample = mimic3base.set_task(task_fn=length_of_stay_prediction_mimic3_fn)
        elif task == "drug_rec":
            mimic3sample = mimic3base.set_task(task_fn=drug_recommendation_mimic3_fn)
        else:
            raise NotImplementedError

        return mimic3sample

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
            self.monitor = "accuracy"
            self.label_key = "drugs"
            return ["accuracy", "f1_macro", "roc_auc_samples"]
