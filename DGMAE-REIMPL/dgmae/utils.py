"""Hàm dùng chung cho bản node-level (GAT) và graph-level (GIN)."""
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_norm(name, dim):
    if name == "batchnorm":
        return nn.BatchNorm1d(dim)
    if name == "layernorm":
        return nn.LayerNorm(dim)
    return nn.Identity()
