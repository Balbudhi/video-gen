import torch
import torchvision
import numpy as np
import os.path as osp
from PIL import Image
import torchvision
import torchvision.transforms as TF


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class CenterCropArr:
    def __init__(self, img_size: int):
        self.img_size = int(img_size)

    def __call__(self, x):
        return center_crop_arr(x, self.img_size)


class StackToTensorWithHFlip:
    def __init__(self):
        self._tt = TF.ToTensor()

    def __call__(self, x):
        return torch.stack([self._tt(x), self._tt(TF.functional.hflip(x))])


class MultiCenterCropArr:
    def __init__(self, crop_sizes):
        self.crop_sizes = [int(s) for s in crop_sizes]

    def __call__(self, x):
        return [center_crop_arr(x, s) for s in self.crop_sizes]


class TenCropFlatten:
    def __init__(self, img_size: int):
        self.ten = TF.TenCrop(int(img_size))

    def __call__(self, crops):
        return [c for crop in crops for c in self.ten(crop)]


class ToTensorStack:
    def __init__(self):
        self._tt = TF.ToTensor()

    def __call__(self, crops):
        return torch.stack([self._tt(c) for c in crops])


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def vae_transforms(split, aug="randcrop", img_size=256):
    t = []
    if split == "train":
        if aug == "randcrop":
            t.append(
                TF.Resize(
                    img_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True
                )
            )
            t.append(TF.RandomCrop(img_size))
        elif aug == "centercrop":
            t.append(CenterCropArr(img_size))
        else:
            raise ValueError(f"Invalid augmentation: {aug}")
        t.append(TF.RandomHorizontalFlip(p=0.5))
    else:
        t.append(CenterCropArr(img_size))

    t.append(TF.ToTensor())

    return TF.Compose(t)


def cached_transforms(aug="tencrop", img_size=256, crop_ranges=[1.05, 1.10]):
    t = []
    if "centercrop" in aug:
        t.append(CenterCropArr(img_size))
        t.append(StackToTensorWithHFlip())
    elif "tencrop" in aug:
        crop_sizes = [int(img_size * crop_range) for crop_range in crop_ranges]
        t.append(MultiCenterCropArr(crop_sizes))
        t.append(TenCropFlatten(img_size))
        t.append(ToTensorStack())
    else:
        raise ValueError(f"Invalid augmentation: {aug}")

    return TF.Compose(t)


class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, root, split="train", aug="randcrop", img_size=256):
        super().__init__(osp.join(root, split))
        if not "cache" in aug:
            self.transform = vae_transforms(split, aug=aug, img_size=img_size)
        else:
            self.transform = cached_transforms(aug=aug, img_size=img_size)
