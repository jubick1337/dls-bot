import torch
import torch.nn.functional as F
from torch import nn

from model.utils import gram_matrix


class ContentLoss(nn.Module):
    def __init__(self, target: torch.Tensor):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(input_tensor, self.target)
        return input_tensor


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        G = gram_matrix(input_tensor)
        self.loss = F.mse_loss(G, self.target)
        return input_tensor
