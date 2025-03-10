import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms
import torchyin
from einops import rearrange, repeat


def l2_normalize(x, dim=None):
    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    return x / norm


def cosine_similarity(input1, input2=None):
    """calculate cosine_similarity among N vectors

    Args:
        input1: (N, L)
        input2: (N, L) or None. when `input2` is None, input2 will be input1

    Return:
        similarity matrix `C` with size $N \times N$, where `C_ij` is the
        cosine_similarity between input1[i, :] and input[j, :]
    """
    assert input1.ndim == 2
    if input2 is not None:
        assert input2.ndim == 2
    input1 = l2_normalize(input1, dim=-1)
    input2 = l2_normalize(input2, dim=-1) if input2 is not None else input1
    return torch.matmul(input1, input2.t())


# $$
# \begin{aligned}
# \mathcal{L}_{\text {con }}= & \frac{1}{N^2} \sum_i^N\left(\sum_{j: y_i=y_j}^N\left(1-\frac{\mathbf{Z}_i \cdot \mathbf{Z}_j}{\left\|\mathbf{Z}_i\right\|\left\|\mathbf{Z}_j\right\|}\right)\right. \\
# & \left.+\sum_{j: y_i\neq y_j}^N \max \left(\frac{\mathbf{Z}_i \cdot \mathbf{Z}_j}{\left\|\mathbf{Z}_i\right\|\left\|\mathbf{Z}_j\right\|}-\alpha, 0\right)\right)
# \end{aligned}
# $$


class BinaryTokenContrastLoss(nn.Module):
    def __init__(self, alpha=0.3, distance="cosine_similarity"):
        super().__init__()
        self.alpha = alpha
        if isinstance(distance, str):
            if distance == "cosine_similarity":
                self.distance = cosine_similarity
        else:
            self.distance = distance

    def forward(self, tokens, labels):
        assert tokens.size(0) == labels.size(0)
        assert tokens.ndim == 2
        similariry_matrix = self.distance(tokens)
        label_matrix = labels[:, None] + labels[None, :]
        loss = torch.where(
            label_matrix != 1,  # lable pairs (0, 0) , (1, 1)
            1 - similariry_matrix,
            torch.maximum(
                similariry_matrix - self.alpha, torch.zeros_like(similariry_matrix)
            ),
        )
        return torch.mean(loss)


class MultiClass_ContrastLoss(nn.Module):
    def __init__(self, alpha=0.3, distance="cosine_similarity"):
        super().__init__()
        self.alpha = alpha
        self.distance = distance
        if distance == "cosine_similarity":
            self.distance_func = cosine_similarity
        elif distance == "l2":
            self.distance_func = lambda x: torch.cdist(x, x)

    def forward(self, tokens, labels):
        tokens = F.normalize(tokens, dim=-1)
        assert tokens.size(0) == labels.size(0)
        assert tokens.ndim == 2
        similariry_matrix = self.distance_func(tokens)
        label_matrix = labels[:, None] - labels[None, :]
        if self.distance == "cosine_similarity":
            loss = torch.where(
                label_matrix != 0,  # lable pairs (0, 0) , (1, 1)
                1 - similariry_matrix,
                torch.maximum(
                    similariry_matrix - self.alpha, torch.zeros_like(similariry_matrix)
                ),
            )
        else:
            loss = torch.where(
                label_matrix != 0,  # lable pairs (0, 0) , (1, 1)
                similariry_matrix,
                torch.maximum(
                    self.alpha - similariry_matrix, torch.zeros_like(similariry_matrix)
                ),
            )
        return torch.mean(loss)


def get_f0_loss(x, pred_f0):
    """
    Assume that the input audio x is with shape (B, 1, 48000). If its length is not equal to 48000,
    you may have to change th frame stride (second).
    """
    pitch = (
        torchyin.estimate(
            x[:, 0, :],
            sample_rate=16000,
            pitch_min=20,
            pitch_max=9000,
            frame_stride=0.01513,  # actually is 0.015625
        )
        / 9000
    )

    loss = F.mse_loss(pred_f0, pitch)
    return loss
