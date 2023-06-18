import random
import logging

import cv2
import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import rgb_to_grayscale
from PIL import Image
import torch
from kornia import morphology
import numpy as np
import torch


# 0: InterpolationMode.NEAREST,
# 2: InterpolationMode.BILINEAR,
# 3: InterpolationMode.BICUBIC,
# 4: InterpolationMode.BOX,
# 5: InterpolationMode.HAMMING,
# 1: InterpolationMode.LANCZOS,
class InterpolationMode():
    NEAREST = 0
    BILINEAR = 2
    BICUBIC = 3
    BOX = 4
    HAMMING = 5
    LANCZOS = 1


logger = logging.getLogger(__name__)


class ResizePad(object):

    def __init__(self, imgH=64, imgW=3072, keep_ratio_with_pad=True):
        self.imgH = imgH
        self.imgW = imgW
        assert keep_ratio_with_pad == True
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, im):

        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(self.imgH) / old_size[1]
        new_size = tuple([int(x * ratio) for x in old_size])
        im = im.resize(new_size, Image.BICUBIC)

        new_im = Image.new("RGB", (self.imgW, self.imgH))
        new_im.paste(im, (0, 0))

        return new_im


class Dilation(torch.nn.Module):

    def __init__(self, kernel=3, device=None):
        super().__init__()
        # self.kernel = kernel
        self.kernel = torch.ones(kernel, kernel)
        if device:
            self.kernel.to(device)

    def forward(self, img):
        # return img.filter(ImageFilter.MaxFilter(self.kernel))
        if len(img.shape) == 4:
            return morphology.dilation(img, self.kernel)

        img = img.unsqueeze(0)
        return morphology.dilation(img, self.kernel).squeeze()

    def __repr__(self):
        return self.__class__.__name__ + f'(kernel={self.kernel})'


class Erosion(torch.nn.Module):

    def __init__(self, kernel=3, device=None):
        super().__init__()
        # self.kernel = kernel
        self.kernel = torch.ones(kernel, kernel)
        if device:
            self.kernel.to(device)

    def forward(self, img):
        # return img.filter(ImageFilter.MinFilter(self.kernel))
        if len(img.shape) == 4:
            return morphology.erosion(img, self.kernel)

        img = img.unsqueeze(0)
        return morphology.erosion(img, self.kernel).squeeze()

    def __repr__(self):
        return self.__class__.__name__ + f'(kernel={self.kernel})'


class Underline(torch.nn.Module):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, img_tensor):
        batch_size = img_tensor.shape[0]
        grayscale = rgb_to_grayscale(img_tensor, num_output_channels=1)

        underline_mask = torch.zeros_like(img_tensor)

        for b in range(batch_size):
            black_pixels = torch.where(grayscale[b] < self.threshold)
            try:
                y1 = int(torch.max(black_pixels[1]))
                x0 = int(torch.min(black_pixels[2]))
                x1 = int(torch.max(black_pixels[2]))
            except Exception as err:
                logger.log(40, err, exc_info=err)
                return img_tensor

            for x in range(x0, x1):
                for y in range(y1, y1 - 3, -1):
                    try:
                        underline_mask[b, :, y, x] = 1.0
                    except Exception as err:
                        logger.log(40, err, exc_info=err)
                        continue

        img_tensor = img_tensor * (1 - underline_mask) + underline_mask * 0.0
        return img_tensor


class RandomInkSpots(torch.nn.Module):
    def __init__(self, ink_spots_num=5, ink_spot_size=5, ink_spot_color=(0, 0, 0)):
        super().__init__()
        self.ink_spots_num = ink_spots_num
        self.ink_spot_size = ink_spot_size
        self.ink_spot_color = ink_spot_color

    def forward(self, img):
        # Convert PyTorch tensor to numpy array
        img = img.permute(1, 2, 0).numpy()

        spot_num = random.randint(1, self.ink_spots_num)

        for _ in range(spot_num):
            # Random position for the center of the ink spot
            center_x = random.uniform(0, img.shape[1])
            center_y = random.uniform(0, img.shape[0])

            # Generate some random points around the center
            points = np.array([[(random.uniform(center_x - self.ink_spot_size, center_x + self.ink_spot_size),
                                 random.uniform(center_y - self.ink_spot_size, center_y + self.ink_spot_size))] for _ in range(5)], dtype=np.int32)

            # Draw the ink spot using OpenCV
            cv2.fillPoly(img, [points], self.ink_spot_color)

        # Convert numpy array back to PyTorch tensor
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img


class KeepOriginal(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img


def build_data_aug(size, mode="train", resnet=False, resizepad=False, device=None):
    if resnet:
        norm_tfm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    else:
        norm_tfm = transforms.Normalize(mean=[0.5], std=[0.5])

    if resizepad:
        resize_tfm = ResizePad(imgH=size[0], imgW=size[1])
    else:
        resize_tfm = transforms.Resize(
            size, interpolation=InterpolationMode.BICUBIC, antialias=True)

    if mode == 'train':
        return transforms.Compose([
            transforms.ToImageTensor(),
            transforms.ConvertImageDtype(dtype=torch.float32),
            transforms.Resize(
                64, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomChoice([
                Underline(),
                RandomInkSpots(),
                KeepOriginal(),
            ]),
            transforms.RandomChoice([
                transforms.RandomRotation(degrees=(-5, 5),
                                          expand=True,
                                          fill=(1, 1, 1)),
                transforms.GaussianBlur(3),
                transforms.Resize(
                    (size[0] // 3, size[1] // 3), InterpolationMode.BICUBIC, antialias=True),
                Dilation(3, device),
                Erosion(3, device),
                KeepOriginal(),

            ]),
            resize_tfm,
            norm_tfm,
            transforms.ToImagePIL()
        ])
