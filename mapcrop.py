"""
data loader for adaptive sampling based on binary maps.
compare randomly cropped image to corresponding binary map area.

I. pseudo code
    A. initialize
        i. grap image crop/binary map
        ii. calculate total binary area w/in crop (sum)
    B. iterate
        i. while (binary map < 50%):
            a. grab new crop
            b. calculate total area

"""

from __future__ import division
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from PIL import Image
import pdb
from SmallScale import SmallScale
import os
import math
import matplotlib
matplotlib.use('Agg')
import sys
import random
import numbers
import types
import collections
import warnings
import cv2
import numpy as np
from PIL import Image
import skimage
from skimage.filters.rank import entropy
import scipy as sp
from scipy.ndimage.morphology import binary_fill_holes
from skimage import data, io, filters, img_as_float
from skimage.filters import threshold_otsu
from skimage import exposure
from skimage.morphology import disk, opening, remove_small_objects
from skimage.transform import resize
from skimage import img_as_bool
import pdb
import sys
from torchvision.transforms import functional as F
try:
    import accimage
except ImportError:
    accimage = None


class MapCrop(object):
    """Crop the given PIL Image at a random location containing at least 25% of the total .

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, tensor, remove_background, crop_params=None, padding=None, pad_if_needed=False, fill=0, padding_mode='constant', thresh=0.1):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.tensor = tensor
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.thresh = thresh
        self.remove_background = remove_background
        self.crop_params = crop_params

        # TODO: here load all image lists by class
        # self.data_folder = '/data1/adamw/HDNeuron/tensors'
        # # classes = ['Debris', 'Dense', 'Diff', 'Spread']
        # self.debris = os.listdir(os.path.join(self.data_folder, 'Debris', 'Debris'))
        # self.dense = os.listdir(os.path.join(self.data_folder, 'Dense', 'Dense'))
        # self.diff = os.listdir(os.path.join(self.data_folder, 'Diff', 'Diff'))
        # self.spread = os.listdir(os.path.join(self.data_folder, 'Spread', 'Spread'))

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    # TODO: take image name in order to load corresponding map tensor...
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.

        """

        # TODO: get image filename, index, or class and pass to Map
        img_thresh = 0
        loop = 0
        if self.crop_params is None: 
            while img_thresh < self.thresh:

                # TODO: load image tensors instead of  processing on the fly
                img_tensor = self.tensor
                # img_tensor = map_tensor(img)
                img_array = np.array(img_tensor).astype(int)
                # print(img_array)
                img_np = np.array(img)
                global cell_body
                if img_array.shape != img_np.shape:
                    img_array = img_as_bool(resize(img_array, img_np.shape))  # .astype(int)
                    img_tensor = Image.fromarray(img_array)

                # compare corresponding binary maps of input crops
                if self.padding is not None:
                    img = F.pad(self.img, self.padding, self.fill, self.padding_mode)
                    img_tensor = F.pad(img_tensor, self.padding, self.fill, self.padding_mode)

                # pad the width if needed
                if self.pad_if_needed and img.size[0] < self.size[1]:
                    img = F.pad(self.img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
                    img_tensor = F.pad(img_tensor, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)

                # pad the height if needed
                if self.pad_if_needed and img.size[1] < self.size[0]:
                    img = F.pad(self.img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
                    img_tensor = F.pad(img_tensor, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

                # deteremine the random image crop parameters for the input img
                i, j, h, w = self.get_params(img, self.size)

                # crop the binary map given the crop parameters 
                crop_tensor = F.crop(img_tensor, i, j, h, w)
                crop_array = np.array(crop_tensor).astype(int)

                total_area = img_array.sum()
                # check to see if the total binary area within the tensor is less than the threshold for image crop
                # if True return cropped image
                if total_area / np.size(crop_array) <= self.thresh:   
                    break

                # caclulate the total binary area of the cropped image and 
                # return the proportion to be compared to the crop threshold
                crop_area = crop_array.sum()
                img_thresh = crop_area/np.size(crop_array)

                loop += 1
                if loop == 2:
                    break

        else:
            i, j, h, w = self.crop_params

        """uncomment to remove background via image binarization"""
        if self.remove_background == True and random.randint(0,3) == 0:  # randomly return background removed image at probability 0.25            
            # create a version of the input image without the flat background
            cell_body = binarize(img, img_array)
            return F.crop(cell_body, i, j, h, w), F.crop(img, i, j, h, w)
        # else:
        return F.crop(img, i, j, h, w), F.crop(img, i, j, h, w), (i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def binarize(img_in, img_array):
    img_out = np.array(img_in)
    img_out[img_array == 0] = 0
    
    return Image.fromarray(img_out)

def map_tensor(img_in):

    """
    :param img_in: input image
    :return:
    """

    img_gray = np.array(img_in)
    img_filt = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_ent = normalize_img(entropy(img_filt, disk(3)))
    img_open = opening(img_ent, disk(3))
    thresh = threshold_otsu(img_open)
    _, img_bw = cv2.threshold(img_open, thresh, 255, cv2.THRESH_BINARY)
    img_fill = binary_fill_holes(img_bw)
    img_final = remove_small_objects(img_fill, 2000)

    return Image.fromarray(img_final)


def normalize_img(img_in):
    """
    :param img_in: image to be normalized
    :return: normalized image
    """
    top = img_in.max()
    bottom = img_in.min()
    norm = ((img_in - bottom) / (top - bottom)) * 255

    return norm.round()
