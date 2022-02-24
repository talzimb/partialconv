import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def calculate_metrics(output, target):
    output_cls = output.argmax(dim=1)
    metrics = {}
    metrics['acc'] = accuracy_score(np.array(target), np.array(output_cls))
    metrics['recall'] = recall_score(np.array(target), np.array(output_cls))
    metrics['precision'] = precision_score(np.array(target), np.array(output_cls))

    return metrics


def attach_to_tensorboard(writer, phase, metrics, epoch):

    for metric in metrics.keys():
        writer.add_scalar(metric + '/' + phase, metrics[metric], epoch)