import torch
from torch import nn


def gram_matrix(input_tensor: torch.Tensor) -> torch.Tensor:
    batch_size, h, w, f_map_num = input_tensor.size()
    features = input_tensor.view(batch_size * h, w * f_map_num)
    G = torch.mm(features, features.t())

    return G.div(batch_size * h * w * f_map_num)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


CONTENT_LAYERS_DEFAULT = ['conv_4']
STYLE_LAYERS_DEFAULT = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
