"""Various utilities."""

import os
import csv
import socket
import datetime

from collections import defaultdict

import torch
import random
import numpy as np

from .consts import NON_BLOCKING


def system_startup(args=None, defs=None):
    """Decide and print GPU / CPU / hostname info."""
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float, non_blocking=NON_BLOCKING)
    print('Currently evaluating -------------------------------:')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    if args is not None:
        print(args)
    if defs is not None:
        print(repr(defs))
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')

    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')

    return setup


def average_dicts(running_stats):
    """Average entries in a list of dictionaries."""
    average_stats = defaultdict(list)
    for stat in running_stats[0]:
        if isinstance(running_stats[0][stat], list):
            for i, _ in enumerate(running_stats[0][stat]):
                average_stats[stat].append(np.mean([stat_dict[stat][i] for stat_dict in running_stats]))
        else:
            average_stats[stat] = np.mean([stat_dict[stat] for stat_dict in running_stats])
    return average_stats


"""Misc."""
def _gradient_matching(poison_grad, target_grad):
    """Compute the blind passenger loss term."""
    matching = 0
    poison_norm = 0
    target_norm = 0

    for pgrad, tgrad in zip(poison_grad, target_grad):
        matching -= (tgrad * pgrad).sum()
        poison_norm += pgrad.pow(2).sum()
        target_norm += tgrad.pow(2).sum()

    matching = matching / poison_norm.sqrt() / target_norm.sqrt()

    return matching


def bypass_last_layer(model):
    """Hacky way of separating features and classification head for many models.

    Patch this function if problems appear.
    """
    layer_cake = list(model.children())
    last_layer = layer_cake[-1]
    headless_model = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())  # this works most of the time all of the time :<
    return headless_model, last_layer



def cw_loss(outputs, intended_classes, clamp=-100):
    """Carlini-Wagner loss for brewing [Liam's version]."""
    top_logits, _ = torch.max(outputs, 1)
    intended_logits = torch.stack([outputs[i, intended_classes[i]] for i in range(outputs.shape[0])])
    difference = torch.clamp(top_logits - intended_logits, min=clamp)
    return torch.mean(difference)

def _label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.shape[0], num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cw_loss2(outputs, intended_classes, confidence=0, clamp=-100):
    """CW variant 2. This is assert-level equivalent."""
    one_hot_labels = _label_to_onehot(intended_classes, num_classes=outputs.shape[1])
    target_logit = (outputs * one_hot_labels).sum(dim=1)
    second_logit, _ = (outputs - outputs * one_hot_labels).max(dim=1)
    cw_indiv = torch.clamp(second_logit - target_logit + confidence, min=clamp)
    return cw_indiv.mean()


def record_results(kettle, brewed_loss, results, args, defs, modelkey, extra_stats=dict()):
    """Save output to a csv table."""
    class_names = kettle.trainset.classes
    stats_clean, stats_rerun, stats_results = results

    def _maybe(stats, param, mean=False):
        """Retrieve stat if it was recorded. Return empty string otherwise."""
        if stats is not None:
            if len(stats[param]) > 0:
                if mean:
                    return np.mean(stats[param])
                else:
                    return stats[param][-1]

        return ''

def automl_bridge(kettle, poison_delta, name, mode='poison-upload', dryrun=False):
    """Transfer data to autoML code. Lazy init due to additional libraries."""
    from .gcloud import automl_interface
    """Send data to autoML."""
    setup = dict(uid=name,
                 project_id='YOUR-PROJECT-ID',
                 multilabel=False,
                 format='.png',
                 bucketname='YOUR-BUCKET-NAME',
                 display_name=name,
                 dataset_id=None,
                 model_id=None,
                 ntrial=1,
                 mode=mode,
                 base_dataset='ImageNet' if kettle.args.dataset == 'ImageNet1k' else kettle.args.dataset,
                 dryrun=dryrun)
    automl_interface(setup, kettle, poison_delta)

def set_random_seed(seed=233):
    """233 = 144 + 89 is my favorite number."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)

def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
