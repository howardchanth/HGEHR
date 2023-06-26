import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from collections import OrderedDict

import logging

import numpy as np

from pyhealth.metrics import binary_metrics_fn, multilabel_metrics_fn, multiclass_metrics_fn

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve, roc_auc_score


def ordered_yaml():
    """
    yaml orderedDict support
    """
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def acc(outputs, targets):
    return np.mean(outputs == targets)


def metrics(outputs, targets, t, prefix="tr"):
    # outputs = outputs.softmax(1).detach().cpu().numpy()
    # preds = outputs.argmax(1)
    # targets = targets.detach().cpu().numpy()
    # precision = precision_score(targets, preds, average=average)
    # recall = recall_score(targets, preds, average=average)
    # f1 = f1_score(targets, preds, average=average)
    # if average == 'binary':
    #     fpr, tpr, thresholds = roc_curve(targets, preds)
    #     aucroc = auc(fpr, tpr)
    # else:
    #     aucroc= roc_auc_score(targets, outputs, multi_class='ovo')
    #     # aucroc = roc_auc_score(targets, outputs[:,1], multi_class='ovo')

    # accuracy = acc(preds, targets)

    if t in ["mort_pred", "readm"]:
        met = binary_metrics_fn(
            targets.detach().cpu().numpy(),
            outputs.softmax(1)[:, 1].detach().cpu().numpy(),
            metrics=["accuracy", "roc_auc", "f1", "pr_auc"]
        )
    elif t == "los":
        met = multiclass_metrics_fn(
            targets.detach().cpu().numpy(),
            outputs.softmax(1).detach().cpu().numpy(),
            metrics=["roc_auc_weighted_ovo", "f1_weighted", "accuracy"]
        )
    elif t == "drug_rec":
        met = multilabel_metrics_fn(
            targets.detach().cpu().numpy(),
            outputs.softmax(1).detach().cpu().numpy(),
            metrics=["roc_auc_samples", "pr_auc_samples", "accuracy", "f1_samples"]
        )
    else:
        raise ValueError

    met = {f"{prefix}_{k}": v for k, v in met.items()}
    return met
    #
    # return {
    #     # f"{prefix}_prec": precision,
    #     # f"{prefix}_recall": recall,
    #     f"{prefix}_accuracy": accuracy,
    #     f"{prefix}_auroc": aucroc,
    #     f"{prefix}_f1": f1
    # }
