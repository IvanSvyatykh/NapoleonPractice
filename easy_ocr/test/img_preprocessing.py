import torch
import numpy as np
import math
from PIL import Image, ImageFile
from torchvision.transforms.v2 import ToTensor

class NormalizePAD(object):
    def __init__(self, max_size, PAD_type="right"):
        self.toTensor = ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = (
                img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
            )

        return Pad_img

def __contrast_grey(img: np.ndarray) -> np.ndarray:
    high = np.percentile(img, 90)
    low = np.percentile(img, 10)
    return (high - low) / (high + low), high, low


def __adjust_contrast_grey(img: np.ndarray, target: float = 0.4) -> np.ndarray:
    contrast, high, low = __contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200.0 / (high - low)
        img = (img - low + 25) * ratio
        img = np.maximum(
            np.full(img.shape, 0), np.minimum(np.full(img.shape, 255), img)
        ).astype(np.uint8)
    return img


def prepare_photo(img: ImageFile,imgH:int=32, imgW:int=100, contrast_adjust: float = 0.0) -> torch.Tensor:
    w, h = img.size
    resized_max_w = imgW
    input_channel = 3 if img.mode == "RGB" else 1
    if input_channel == 3 :
        img = img.convert("L")
        input_channel =1
    transform = NormalizePAD((input_channel,imgH, resized_max_w))
    if contrast_adjust > 0:
        img = np.array(img.convert("L"))
        img = __adjust_contrast_grey(img, target=contrast_adjust)
        img = Image.fromarray(img, "L")

    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = math.ceil(imgH * ratio)

    resized_image = img.resize((resized_w, imgH), Image.BICUBIC)
    resized_image = transform(resized_image)
    return resized_image
