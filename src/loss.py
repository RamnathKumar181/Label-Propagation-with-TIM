import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def get_entropy(probs):
    probs = torch.exp(probs)
    q_ent = - (probs.mean(0) * torch.log(probs.mean(0) + 1e-12)).sum()
    return q_ent

def get_cond_entropy(probs):
    probs = torch.exp(probs)
    q_cond_ent = - (probs * torch.log(probs + 1e-12)).sum(1).mean(0, keepdim=True)
    return q_cond_ent

def get_cross_entropy_loss(y_pred, y_true):
    ce_loss = nn.CrossEntropyLoss()
    return ce_loss(y_pred, y_true)
