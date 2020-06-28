import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from model.utils import GeneratorResNet


class SRRes:
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = GeneratorResNet().to(self._device).eval()
        self._net.load_state_dict(torch.load('./generator_5.pth', map_location=self._device))
        self._loader = transforms.Compose([
            # transforms.Resize(512, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                 np.array([0.229, 0.224, 0.225]))
        ])
        self._unloader = transforms.ToPILImage()

    def _image_loader(self, image_name: str) -> torch.Tensor:
        image = Image.open(image_name)
        image = self._loader(image).unsqueeze(0)
        return image.to(self._device, torch.float)

    def transform(self, input_image: str) -> Image.Image:
        image = self._image_loader(input_image)
        res = self._net(image)
        with torch.no_grad():
            return self._unloader(res.squeeze_(0))
