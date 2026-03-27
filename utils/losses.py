import torch
import torch.nn.functional as F


def d_loss(x, y, type="cosine"):
    if type == "cosine":
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return 1 - (x * y).sum(dim=-1).mean()

    elif type == "l2":
        return ((x - y) ** 2).mean()

    else:
        raise ValueError(f"Unknown loss type: {type}")