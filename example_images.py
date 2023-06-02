import torch
from CustomDataset import CustomDataset 
from torchvision.utils import save_image
import os 
import pdb
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from ImbalancedSampler import ImbalancedDatasetSampler
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None

class SmallScale(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        height, width = img.size
        if height < self.size or width < self.size:
            return resize(img, self.size, self.interpolation)
        else:
            return img


def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


base_path = '/media/adam/dc156fa0-1275-46c2-962c-bc8c9fcf1cb0/ucr_data/data1/HDNeuron/GAN_imgs'
save_path = '/media/adam/dc156fa0-1275-46c2-962c-bc8c9fcf1cb0/ucr_data/data1/contrastive_learning/MPL_save'


img_path = datasets.ImageFolder(base_path, transforms.Compose([SmallScale(128),
                                                   transforms.Grayscale(),
                                                   transforms.CenterCrop(128),
                                                   # transforms.GaussianBlur(3),
                                                   # transforms.RandomRotation(180),
                                                   transforms.ColorJitter(brightness=0.5, contrast=0.75),
                                                   transforms.ToTensor()]))

img_set = DataLoader(img_path, 
					 batch_size=100,
                     sampler=ImbalancedDatasetSampler(img_path),
                     shuffle=False)

for i, (img, target) in enumerate(img_set):


    save_image(img, os.path.join(save_path, 'Color_jitter.png'), nrow=10, normalize=True)
    break

# pdb.set_trace()

