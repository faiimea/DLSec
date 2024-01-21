"""Implement victim behavior, for single-victim, ensemble and stuff."""
import torch

from .victim_single import _VictimSingle

def Victim(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    return _VictimSingle(args, setup)


from ..hyperparameters import training_strategy
__all__ = ['Victim', 'training_strategy']
