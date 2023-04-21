import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from collections import OrderedDict

import logging

import numpy as np

import torch
# from torch_scatter import scatter_mean, scatter_max

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
    return np.mean(outputs.detach().cpu().numpy().argmax(axis=1) == targets.detach().cpu().numpy())

def metrics(outputs, targets, average='binary'):
    preds = outputs.argmax(1)
    precision = precision_score(targets, preds, average=average)
    recall = recall_score(targets, preds, average=average)
    f1 = f1_score(targets, preds, average=average)
    if average == 'binary':
        fpr, tpr, thresholds = roc_curve(targets, preds)
        aucroc = auc(fpr, tpr)
    else:
        aucroc= roc_auc_score(targets, outputs, multi_class='ovo')
        # aucroc = roc_auc_score(targets, outputs[:,1], multi_class='ovo')
    return precision, recall, f1, aucroc

def get_logger():
    logger_name = 'main-logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

# def readout(x, batch):
#     x_mean = scatter_mean(x, batch, dim=0)
#     x_max, _ = scatter_max(x, batch, dim=0)
#     return torch.cat((x_mean, x_max), dim=-1)