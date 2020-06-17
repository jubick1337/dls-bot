import copy
from typing import List

import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn

from model.losses import ContentLoss, StyleLoss
from model.utils import CONTENT_LAYERS_DEFAULT, STYLE_LAYERS_DEFAULT, Normalization


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


class NST:
    def __init__(self, image_size: int):
        self._image_size = image_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loader = transforms.Compose([
            transforms.Resize(self._image_size),
            transforms.CenterCrop(self._image_size),
            transforms.ToTensor()])
        self._cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self._device)
        self._cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self._device)
        self._cnn = models.vgg19(pretrained=True).features.to(self._device).eval()

    def transform(self, content_image: Image, style_image: Image):
        content_image = self._loader(content_image)
        style_image = self._loader(style_image)
        return self._run_style_transfer(content_image, style_image, content_image.clone())

    def _image_loader(self, image_name) -> torch.Tensor:
        image = Image.open(image_name)
        image = self._loader(image).unsqueeze(0)
        return image.to(self._device, torch.float)

    def _run_style_transfer(self, content_image: torch.Tensor, style_image: torch.Tensor, input_image: torch.Tensor,
                            num_steps=500,
                            style_weight=100000, content_weight=1) -> torch.Tensor:
        """Run the style transfer."""
        model, style_losses, content_losses = self._get_style_model_and_losses(content_image, style_image)
        optimizer = get_input_optimizer(input_image)
        run = [0]
        while run[0] <= num_steps:

            def closure():
                input_image.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(input_image)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    return style_score + content_score

            optimizer.step(closure)
        input_image.data.clamp_(0, 1)

        return input_image

    def _get_style_model_and_losses(self, content_img: torch.Tensor, style_img: torch.Tensor,
                                    content_layers=CONTENT_LAYERS_DEFAULT,
                                    style_layers=STYLE_LAYERS_DEFAULT) -> (nn.Sequential, List, List):
        cnn = copy.deepcopy(self._cnn)

        normalization = Normalization(self._cnn_normalization_mean, self._cnn_normalization_std).to(self._device)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses
