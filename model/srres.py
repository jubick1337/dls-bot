import torch
from PIL import Image
from torchvision.transforms import transforms

from model.utils import GeneratorResNet, CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD


class SRRes:
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = GeneratorResNet().to(self._device).eval()
        self._net.load_state_dict(torch.load('./generator_5.pth', map_location=self._device))
        self._loader = transforms.Compose([
            transforms.Resize(512, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(CNN_NORMALIZATION_MEAN,
                                 CNN_NORMALIZATION_STD)
        ])

    def transform(self, input_image: str) -> torch.Tensor:
        image = self._image_loader(input_image)
        return self._net(image)

    def _image_loader(self, image_name: str) -> torch.Tensor:
        image = Image.open(image_name)
        image = self._loader(image).unsqueeze(0)
        return image.to(self._device, torch.float)
