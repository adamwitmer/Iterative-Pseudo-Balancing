import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
import numpy as np
try:
    import accimage
except ImportError:
    accimage = None
import pdb
from mapcrop import MapCrop  # , MultiscaleMapCrop
from torch.nn import functional as F
import torch
import pdb

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.txt']


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


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class MultiscaleDataset(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, img_root, crop_size, thresh, n_scales, remove_background, crop_params=None, transform=None, target_transform=None,  # tensor_root
                 loader=default_loader, normalize=True):
        classes, class_to_idx = find_classes(img_root)
        imgs = make_dataset(img_root, class_to_idx)
        # tensors = make_dataset(tensor_root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + img_root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.thresh = thresh
        self.img_root = img_root
        # self.tensor_root = tensor_root
        # self.tensors = tensors
        self.crop_params = crop_params
        self.remove_background=remove_background
        self.crop_size = crop_size
        self.imgs = imgs
        self.normalize = normalize 
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.n_scales = n_scales
        self.loader = loader
        self.tensor_folder = '/media/adam/dc156fa0-1275-46c2-962c-bc8c9fcf1cb0/ucr_data/data1/contrastive_learning/dataset/SimCLR_datasets/tensors/all_tensors'

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_path, target = self.imgs[index]
        img_name = os.path.basename(img_path)[:-4]
        if self.crop_params is not None:
            crop_indices = self.crop_params[index]
        # img_name = img_path[img_path.find(self.classes[target]):-4]
        # print(img_name)
        # pdb.set_trace()
        tensor_path = os.path.join(self.tensor_folder, img_name + '.txt')
        # tensor_path, target = self.tensors[index]
        img_sample = Image.open(img_path).convert('L')
        tensor_sample = Image.fromarray(np.loadtxt(tensor_path))
        img = SmallScale(self.crop_size)(img_sample)
        tensor_sample = SmallScale(self.crop_size)(tensor_sample)
        if self.crop_params is None:
            img_bin, img_crop, crop_indices = MapCrop(self.crop_size, tensor=tensor_sample, thresh=self.thresh, remove_background=self.remove_background, crop_params=None)(img)  # Random binarization (0.25)
        else:
            img_bin, img_crop, _ = MapCrop(self.crop_size, tensor=tensor_sample, thresh=self.thresh, remove_background=self.remove_background, crop_params=crop_indices)(img)  # Random binarization (0.25)        img_gray = transforms.Grayscale()(img_crop)
        img_gray = transforms.Grayscale()(img_crop)
        img_bin = transforms.Grayscale()(img_bin)

        img = transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.75)], 0.25)(img_bin)
        img = transforms.RandomApply([transforms.GaussianBlur(3)], 0.25)(img)
        img = transforms.RandomApply([transforms.RandomRotation(180)], 0.25)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.RandomVerticalFlip()(img)
        img_center = transforms.CenterCrop(self.crop_size//2)(img)
        img_center_center = transforms.CenterCrop(self.crop_size//4)(img_center)
        img = transforms.ToTensor()(img)
        img_center = transforms.ToTensor()(img_center)
        img_center_center = transforms.ToTensor()(img_center_center)

        img2 = transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.75)], 0.25)(img_bin)
        img2 = transforms.RandomApply([transforms.GaussianBlur(3)], 0.25)(img2)
        img2 = transforms.RandomApply([transforms.RandomRotation(180)], 0.25)(img2)
        img2 = transforms.RandomHorizontalFlip()(img2)
        img2 = transforms.RandomVerticalFlip()(img2)
        img2_center = transforms.CenterCrop(self.crop_size//2)(img2)
        img2_center_center = transforms.CenterCrop(self.crop_size//4)(img2_center)
        img2 = transforms.ToTensor()(img2)
        img2_center = transforms.ToTensor()(img2_center)
        img2_center_center = transforms.ToTensor()(img2_center_center)

        img_gray = transforms.RandomHorizontalFlip()(img_gray)
        img_gray = transforms.RandomVerticalFlip()(img_gray)
        img_gray_center = transforms.CenterCrop(self.crop_size//2)(img_gray)
        img_gray_center_center = transforms.CenterCrop(self.crop_size//4)(img_gray_center)
        img_gray = transforms.ToTensor()(img_gray)
        img_gray_center = transforms.ToTensor()(img_gray_center)
        img_gray_center_center = transforms.ToTensor()(img_gray_center_center)
        # if self.normalize:
        #     img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        if self.normalize:
            transforms.Normalize(mean=[0.485], std=[0.229])(img)
            transforms.Normalize(mean=[0.485], std=[0.229])(img2)
            transforms.Normalize(mean=[0.485], std=[0.229])(img_gray)

            transforms.Normalize(mean=[0.485], std=[0.229])(img_center)
            transforms.Normalize(mean=[0.485], std=[0.229])(img2_center)
            transforms.Normalize(mean=[0.485], std=[0.229])(img_gray_center)

            transforms.Normalize(mean=[0.485], std=[0.229])(img_center_center)
            transforms.Normalize(mean=[0.485], std=[0.229])(img2_center_center)
            transforms.Normalize(mean=[0.485], std=[0.229])(img_gray_center_center)


        pad = (self.crop_size//4, self.crop_size//4, self.crop_size//4, self.crop_size//4)
        pad2 = ((self.crop_size-(self.crop_size//4))//2, (self.crop_size-(self.crop_size//4))//2, (self.crop_size-(self.crop_size//4))//2, (self.crop_size-(self.crop_size//4))//2)

        img_center = F.pad(img_center, pad, "constant", 0)
        img2_center = F.pad(img2_center, pad, "constant", 0)
        img_gray_center = F.pad(img_gray_center, pad, "constant", 0)

        img_center_center = F.pad(img_center_center, pad2, "constant", 0)
        img2_center_center = F.pad(img2_center_center, pad2, "constant", 0)
        img_gray_center_center = F.pad(img_gray_center_center, pad2, "constant", 0)


        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        # pdb.set_trace()
        if self.n_scales == 3:
            return torch.cat((img, img_center, img_center_center), 0), torch.cat((img2, img2_center, img2_center_center), 0), torch.cat((img_gray, img_gray_center, img_gray_center_center), 0), index, target, crop_indices
        else:
            return torch.cat((img, img_center), 0), torch.cat((img2, img2_center), 0), torch.cat((img_gray, img_gray_center), 0), index, target, crop_indices

    def __len__(self):
        return len(self.imgs)


class MultiscaleDatasetTest(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, img_root, crop_size, thresh, n_scales, transform=None, target_transform=None,  # tensor_root
                 loader=default_loader, normalize=True):
        classes, class_to_idx = find_classes(img_root)
        imgs = make_dataset(img_root, class_to_idx)
        # tensors = make_dataset(tensor_root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + img_root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.thresh = thresh
        self.img_root = img_root
        # self.tensor_root = tensor_root
        # self.tensors = tensors
        self.crop_size = crop_size
        self.imgs = imgs
        self.normalize = normalize 
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.n_scales = n_scales
        self.loader = loader
        self.tensor_folder = '/media/adam/dc156fa0-1275-46c2-962c-bc8c9fcf1cb0/ucr_data/data1/contrastive_learning/dataset/SimCLR_datasets/tensors/all_tensors'

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_path, target = self.imgs[index]
        img_name = os.path.basename(img_path)[:-4]
        # img_name = img_path[img_path.find(self.classes[target]):-4]
        # print(img_name)
        # pdb.set_trace()
        tensor_path = os.path.join(self.tensor_folder, img_name + '.txt')
        # tensor_path, target = self.tensors[index]
        img_sample = Image.open(img_path).convert('L')
        # tensor_sample = Image.fromarray(np.loadtxt(tensor_path))
        img = SmallScale(self.crop_size)(img_sample)
        # tensor_sample = SmallScale(self.crop_size)(tensor_sample)
        # img_crop = MapCrop(self.crop_size, tensor=tensor_sample, thresh=self.thresh)(img)  # Random binarization (0.25)
        img_crop = transforms.CenterCrop(self.crop_size)(img)
        img_gray = transforms.Grayscale()(img_crop)
        # img_gray_down = transforms.Grayscale()(img_crop_down)

        # img = transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.75)], 0.25)(img_gray)
        # img = transforms.RandomApply([transforms.GaussianBlur(3)], 0.25)(img)
        # img = transforms.RandomApply([transforms.RandomRotation(180)], 0.25)(img)
        # img = transforms.RandomHorizontalFlip()(img)
        # img = transforms.RandomVerticalFlip()(img)
        img_center = transforms.CenterCrop(self.crop_size//2)(img_gray)
        img_center_center = transforms.CenterCrop(self.crop_size//4)(img_center)
        img = transforms.ToTensor()(img_gray)
        img_center = transforms.ToTensor()(img_center)
        img_center_center = transforms.ToTensor()(img_center_center)

        # img2 = transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.75)], 0.25)(img_gray)
        # img2 = transforms.RandomApply([transforms.GaussianBlur(3)], 0.25)(img2)
        # img2 = transforms.RandomApply([transforms.RandomRotation(180)], 0.25)(img2)
        # img2 = transforms.RandomHorizontalFlip()(img2)
        # img2 = transforms.RandomVerticalFlip()(img2)
        # img2_center = transforms.CenterCrop(64)(img2)
        # img2 = transforms.ToTensor()(img2)
        # img2_center = transforms.ToTensor()(img2_center)

        # img_gray = transforms.RandomHorizontalFlip()(img_gray)
        # img_gray = transforms.RandomVerticalFlip()(img_gray)
        # img_gray_center = transforms.CenterCrop(64)(img_gray)
        # img_gray = transforms.ToTensor()(img_gray)
        # img_gray_center = transforms.ToTensor()(img_gray_center)
        # if self.normalize:
        #     img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        if self.normalize:
            transforms.Normalize(mean=[0.485], std=[0.229])(img)
            # transforms.Normalize(mean=[0.485], std=[0.229])(img2)
            # transforms.Normalize(mean=[0.485], std=[0.229])(img_gray)

            transforms.Normalize(mean=[0.485], std=[0.229])(img_center)
            transforms.Normalize(mean=[0.485], std=[0.229])(img_center_center)

            # transforms.Normalize(mean=[0.485], std=[0.229])(img2_center)
            # transforms.Normalize(mean=[0.485], std=[0.229])(img_gray_center)

        pad = (self.crop_size//4, self.crop_size//4, self.crop_size//4, self.crop_size//4)
        pad2 = ((self.crop_size-(self.crop_size//4))//2, (self.crop_size-(self.crop_size//4))//2, (self.crop_size-(self.crop_size//4))//2, (self.crop_size-(self.crop_size//4))//2)

        img_center = F.pad(img_center, pad, "constant", 0)
        img_center_center = F.pad(img_center_center, pad2, "constant", 0)
        # img2_center = F.pad(img2_center, pad, "constant", 0)
        # img_gray_center = F.pad(img_gray_center, pad, "constant", 0)
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        # pdb.set_trace()
        if self.n_scales == 3:
            return torch.cat((img, img_center, img_center_center), 0), target  #  torch.cat((img2, img2_center), 0), torch.cat((img_gray, img_gray_center), 0), index, 
        else:
            return torch.cat((img, img_center), 0), target  #  torch.cat((img2, img2_center), 0), torch.cat((img_gray, img_gray_center), 0), index, 


    def __len__(self):
        return len(self.imgs)



