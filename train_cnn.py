"""
This program trains a CNN on image patches from a stem cell microscopy dataset with 4 classes.
CNN's are trained with and without the addition of generated image patches

"""

from __future__ import division
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data.sampler as sampler
import torch.nn.functional as F
from torch.nn import DataParallel
from torchvision import datasets, models
# from tensorboardX import SummaryWriter
from torch.autograd import Variable
# import skimage.feature as sk
# from sklearn.cluster import KMeans
import scipy.misc
import argparse
# from CustomDataset import CustomDataset, CustomDataset224
from ImbalancedSampler import ImbalancedDatasetSampler
import time
from datetime import timedelta
from sklearn.metrics import confusion_matrix
import pdb
import sys
import os
# import cv2
from PIL import Image, ImageFilter
import numpy as np
# from setproctitle import setproctitle
# from tessa import dictionary, sfta, chog
import matplotlib
import pickle
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet
# from HESCnet import HESCnet
# import vgg_models
# from torchvision.models import Vgg19_bn
# from vgg19_128 import Vgg19 
# from ShallowNet import ShallowNet
import cls_hrnet as HRNet
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
try:
    import accimage
except ImportError:
    accimage = None
from torchmetrics.classification import MulticlassAUROC

# setproctitle('Adam W - cnn')

parser = argparse.ArgumentParser()
parser.add_argument('--d_in', type=int, default=674, help='length of texture feature vector used to train NN')
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--train_path', type=str, default='/media/adam/dc156fa0-1275-46c2-962c-bc8c9fcf1cb0/ucr_data/data1/contrastive_learning/dataset/SimCLR_datasets')  # /data1/adamw/HDNeuron')  # /GAN_imgs')
parser.add_argument('--save_path', type=str, default='/media/adam/dc156fa0-1275-46c2-962c-bc8c9fcf1cb0/ucr_data/data1/contrastive_learning/MPL_save/HRNet')  # /data1/adamw/HDNeuron/GAN_imgs_four_classes_128/save/save')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs of training')
parser.add_argument('--crop_thresh', type=float, default=0.5, help='crop threshold for image patches in CustomDataset()')
parser.add_argument('--lr_adam', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--lr_sgd', type=float, default=5e-3, help='sgd: learning rate')
parser.add_argument('--momentum_sgd', type=float, default=0.9, help='sgd: momentum')
parser.add_argument('--wd_sgd', type=float, default=0.0001, help='sgd: weight decay - L2 regularization')
parser.add_argument('--save_int', type=int, default=10, help='network save interval')
parser.add_argument('--gan_batches', type=int, default=0, help='number of gan batches to add during training')
parser.add_argument('--n_folds', type=int, default=5)
parser.add_argument('--evaluate', type=bool, default=True)
opt = parser.parse_args()


def calculate_metrics(confmat, n_class):

    tpr_cm = np.zeros(n_class)
    tnr_cm = np.zeros(n_class)
    f1_cm = np.zeros(n_class)
    acc_cm = np.zeros(n_class)
    # determine accuracy metrics
    for y in range(n_class):
        # true positive rate
        pos = np.sum(confmat[:, y])  # number of instances in the positive class
        tp = confmat[y, y]  # correctly classified positive incidents
        fp = np.sum(confmat[y, :]) - tp  # incorrectly classified negative instances
        tpr = tp / pos  # true positive classification rate
        tpr_cm[y] = tpr
        # true negative rate
        tn = np.trace(confmat) - tp  # correctly classified negative instances
        tnr = tn / (tn + fp)  # true negative rate
        tnr_cm[y] = tnr
        # f1 score
        ppv = tp / (tp + fp)  # positve prediction value
        f1 = 2 * ((ppv * tpr) / (ppv + tpr))  # f1 score
        f1_cm[y] = f1
        tot = np.sum(confmat[y, :])
        acc = tp / tot
        acc_cm[y] = acc
        # dice similarity coefficient (dsc)
        # dsc = 2 * tp / (2 * tp + fp + tn)
        # dsc_cm[fold, y] = dsc

    return tpr_cm, tnr_cm, f1_cm, acc_cm


def modify_resnet_model(model, *, cifar_stem=True, v1=True):
    """Modifies some layers of a given torchvision resnet model to
    match the one used for the CIFAR-10 experiments in the SimCLR paper.

    Parameters
    ----------
    model : ResNet
        Instance of a torchvision ResNet model.
    cifar_stem : bool
        If True, adapt the network stem to handle the smaller CIFAR images, following
        the SimCLR paper. Specifically, use a smaller 3x3 kernel and 1x1 strides in the
        first convolution and remove the max pooling layer.
    v1 : bool
        If True, modify some convolution layers to follow the resnet specification of the
        original paper (v1). torchvision's resnet is v1.5 so to revert to v1 we switch the
        strides between the first 1x1 and following 3x3 convolution on the first bottleneck
        block of each of the 2nd, 3rd and 4th layers.

    Returns
    -------
    Modified ResNet model.
    """
    assert isinstance(model, ResNet), "model must be a ResNet instance"
    if cifar_stem:
        conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
        model.conv1 = conv1
        model.maxpool = nn.Identity()
    if v1:
        for l in range(2, 5):
            layer = getattr(model, "layer{}".format(l))
            block = list(layer.children())[0]
            if isinstance(block, Bottleneck):
                assert block.conv1.kernel_size == (1, 1) and block.conv1.stride == (
                    1,
                    1,
                )
                assert block.conv2.kernel_size == (3, 3) and block.conv2.stride == (
                    2,
                    2,
                )
                assert block.conv2.dilation == (
                    1,
                    1,
                ), "Currently, only models with dilation=1 are supported"
                block.conv1.stride = (2, 2)
                block.conv2.stride = (1, 1)
    return model


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
   

def class_weights(data_set):

    class1 = class2 = class3 = class4 = 0
    for point, (x, y) in enumerate(data_set):
        sys.stdout.write('\rGathering class weights...{}/{}'.format(point, len(data_set)))
        class1 += sum(y == 0)
        class2 += sum(y == 1)
        class3 += sum(y == 2)
        class4 += sum(y == 3)
    weights = torch.Tensor([class1, class2, class3, class4])
    return weights.max() / weights


def class_numbers(data_set):

    class1 = class2 = class3 = class4 = 0
    for point, (x, y) in enumerate(data_set):
        sys.stdout.write('\rGathering class numbers...{}/{}'.format(point, len(data_set)))
        class1 += sum(y == 0)
        class2 += sum(y == 1)
        class3 += sum(y == 2)
        class4 += sum(y == 3)
    weights = torch.Tensor([class1, class2, class3, class4])
    return weights


def normalize_vector(vec):

    top = vec.max()
    bottom = vec.min()
    norm = ((vec-bottom)/(top-bottom))

    return norm  # .round()


def normalize_img(img):

    top = img.max()
    bottom = img.min()
    norm = ((img-bottom)/(top-bottom)) * 255

    return norm.round()


data_folders = ['four_class']  

n_folds = 5

seeds = [1046, 6401, 51, 200589, 50098]  # 1046, 6401, 51]  #, ]  #

normalize = transforms.Normalize(mean=[0.485], std=[0.229])

networks = ['HRNet']  #   'resnet18',  'vgg19']   # 'vgg13', 'vgg16', 'resnet50'

fold_tpr = []
fold_tnr = []
fold_f1 = []
fold_acc = []
fold_roc = []
for network in networks:

    for folder in data_folders:


        for fold in range(opt.n_folds):
            if folder.startswith('temporal'):
                opt.n_classes = 2
           
            new_data_path = os.path.join(opt.train_path, folder, 'fold_{}'.format(fold))

            train_data = datasets.ImageFolder(os.path.join(new_data_path, 'test'), transforms.Compose([SmallScale(opt.crop_size),
                                                                                  transforms.Grayscale(),
                                                                                  transforms.RandomCrop(opt.crop_size),
                                                                                  transforms.RandomHorizontalFlip(),
                                                                                  transforms.RandomVerticalFlip(),
                                                                                  transforms.ToTensor(),
                                                                                  transforms.Normalize(mean=[0.485], std=[0.229])]))

            valid_data = datasets.ImageFolder(os.path.join(new_data_path, 'valid'),
                                              transforms.Compose([SmallScale(opt.crop_size),
                                              transforms.Grayscale(),
                                              transforms.RandomCrop(opt.crop_size),
                                              # transforms.RandomHorizontalFlip(),
                                              # transforms.RandomVerticalFlip(),
                                              transforms.ToTensor(),
                                              normalize]))
            
            """if performing on fly data split"""
            # num_train = len(train_data)
            # indices = list(range(num_train))
            # split = int(np.floor(0.2 * num_train))  # 0.8 --> 20% train data; 0.2 --> 80% train data
            # np.random.seed(seeds[fold])
            # np.random.shuffle(indices)
            #
            # train_idx, test_idx = indices[split:], indices[:split]
            # train_sampler = sampler.SubsetRandomSampler(train_idx)
            # train_loader = torch.utils.data.DataLoader(train_data,
            #                                            sampler=train_sampler,
            #                                            batch_size=64,
            #                                            shuffle=False)
            #
            
            train_loader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=opt.batch_size,
                                                       sampler=ImbalancedDatasetSampler(train_data),
                                                       shuffle=False)

            valid_loader = torch.utils.data.DataLoader(valid_data,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False)

            # Define model, loss function, optimizer, other parameters
            opt.d_in = 64  
            
            if network == 'vgg13':
                model_name = network
                CNN_model = DataParallel(vgg_models.vgg13_bn(num_classes=opt.n_classes).cuda())
            elif network == 'vgg16':
                model_name = network
                CNN_model = DataParallel(vgg_models.vgg16_bn(num_classes=opt.n_classes).cuda())
            elif network == 'vgg19':
                model_name = network
                CNN_model = Vgg19(num_classes=opt.n_classes).cuda()
            elif network == 'ShallowNet':
                model_name = network
                CNN_model = DataParallel(ShallowNet(num_classes=opt.n_classes).cuda())
            elif network == 'resnet18':
                model_name = network
                CNN_model = models.resnet18(num_classes=opt.n_classes)
                CNN_model = modify_resnet_model(CNN_model)
                # CNN_model.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
                CNN_model = DataParallel(CNN_model.cuda())
                # CNN_model.cuda()
            elif network == 'resnet50':
                model_name = network
                CNN_model = models.resnet50(num_classes=opt.n_classes, pretrained=False)
                CNN_model = modify_resnet_model(CNN_model)
                # CNN_model.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
                CNN_model = DataParallel(CNN_model.cuda())
            elif network == 'HRNet':
                model_name = network
                CNN_model = HRNet.get_cls_net()
                CNN_model = DataParallel(CNN_model.cuda())
                # CNN_model.fc = nn.Identity()
            print(f'Params: {sum(p.numel() for p in CNN_model.parameters())/1e6:.2f}M')
           

            """Load Model"""
            
            print("Model Parameters Reset.")

            """Load pretrained model"""
            
            loss_func = torch.nn.CrossEntropyLoss() 
          
            optim_func = torch.optim.SGD(CNN_model.parameters(), lr=opt.lr_sgd,
                                         momentum=opt.momentum_sgd, weight_decay=opt.wd_sgd)

            Tensor = torch.cuda.FloatTensor
            roc_auc = MulticlassAUROC(num_classes=opt.n_classes, average='none')

            # create unique save folder
            save_folder = os.path.join(opt.save_path, 'fold_{}'.format(fold))
           
            os.makedirs(save_folder, exist_ok=True)

            # initialize training monitoring
            train_losses = []
            train_acc = []
            valid_losses = []
            valid_acc = []
            t = time.time()
            # opt.n_epoch = 100
            for epoch in range(opt.n_epoch):
                # epoch = epoch + 100
                if epoch > 0 and epoch % 100 == 0:
                    lr = opt.lr_sgd/10
                    optim_func = torch.optim.SGD(CNN_model.parameters(), lr=lr, momentum=opt.momentum_sgd,
                                                 weight_decay=opt.wd_sgd)
                    print('SGD lr reduced to: {}'. format(lr))

                # initialize batch loss
                train_loss = 0
                total_correct = 0
                count = 0

                # train model
                CNN_model.train()
                for i, (imgs, labels) in enumerate(train_loader):

                    count += len(labels)
                   
                    # train model --> forward pass, compute BCE loss, compute gradient, optimizer step
                    optim_func.zero_grad()
                    output = CNN_model.forward(imgs.cuda())
                    loss = loss_func(output, labels.cuda())
                    loss.backward()
                    optim_func.step()

                    # gather data
                    train_loss += loss.item()
                    # predictions = np.argmax(output.data.cpu().numpy(), axis=1)
                    predictions = output.argmax(dim=1)
                    correct = predictions.eq(labels.cuda()).sum().item()
                    total_correct += correct

                    # change to output training accuracy and loss parameters, fix 'Train time' output
                    sys.stdout.write(
                        '\rTRAINING {}: Fold: {}, Epoch: {}/{}; Progress: {}%; Train time: {}; Train Loss: {:0.4f}, Train Acc. {:0.4f}%'
                                     .format(folder, fold, epoch + 1, opt.n_epoch, round((i / len(train_loader)) * 100),
                                             str(timedelta(seconds=time.time() - t)), loss.item(), correct/len(labels)
                                             )
                                     )

                train_losses.append(train_loss/len(train_loader))
                train_acc.append(total_correct/count)

               

                # test model
                count = 0
                total_correct = 0
                valid_loss = 0
                CNN_model.eval()
                all_predictions = []
                all_targets = []
                probs = []
                all_labels = []
                for j, (valid_imgs, valid_labels) in enumerate(valid_loader):

                    count += len(valid_labels)

                    output = CNN_model.forward(valid_imgs.cuda())
                    loss = loss_func(output, valid_labels.cuda())

                   
                    valid_loss += loss.item()
                    predictions = output.argmax(dim=1)
                    correct = predictions.eq(valid_labels.cuda()).sum().item()
                    total_correct += correct

                    if epoch + 1 == opt.n_epoch:
                        all_predictions.extend(predictions.cpu().numpy())
                        all_targets.extend(valid_labels.numpy())
                        probs.extend(F.softmax(output, dim=1).detach().cpu())  # .numpy())

                    sys.stdout.write(
                        '\rTESTING {}: Epoch: {}/{}; Progress: {}%; Train time: {}; Valid Loss: {:0.4f}; Valid Acc. {:0.4f}%'
                            .format(folder, epoch + 1, opt.n_epoch, round((j / len(valid_loader)) * 100),
                                    str(timedelta(seconds=time.time() - t)), loss.item(), correct/len(valid_labels)
                                    )
                    )
                if epoch + 1 == opt.n_epoch:
                    confmat = confusion_matrix(all_targets, all_predictions)
                    out_metrics = calculate_metrics(confmat, n_class=opt.n_classes)
                    print(out_metrics)
                    np.savetxt(os.path.join(save_folder, 'validation_metrics.csv'), out_metrics, fmt='%.4f', delimiter=',')

                valid_losses.append(valid_loss / len(valid_loader))
                valid_acc.append(total_correct / count)

            roc_score = roc_auc(torch.cat(probs).reshape(-1, opt.n_classes), torch.tensor(all_targets))
            np.savetxt(os.path.join(save_folder, 'train_losses.csv'), np.array(train_losses), fmt='%.4f', delimiter=',')
            np.savetxt(os.path.join(save_folder, 'train_acc.csv'), np.array(train_acc), fmt='%.4f', delimiter=',')
            np.savetxt(os.path.join(save_folder, 'valid_losses.csv'), np.array(valid_losses), fmt='%.4f', delimiter=',')
            np.savetxt(os.path.join(save_folder, 'valid_acc.csv'), np.array(valid_acc), fmt='%.4f', delimiter=',')
            np.savetxt(os.path.join(save_folder, "roc_auc.csv"), roc_score, fmt='%.4f', delimiter=',')

            fold_tpr.append(out_metrics[0])
            fold_tnr.append(out_metrics[1])
            fold_f1.append(out_metrics[2])
            fold_acc.append(out_metrics[3])
            fold_roc.append(roc_score) 

            torch.save(CNN_model.state_dict(), os.path.join(save_folder, 'cnn_model.pth.tar'))

        pdb.set_trace()

        np.savetxt(os.path.join(opt.save_path, 'tpr_fold.csv'), np.array([fold_tpr.mean(0), fold_tpr.std(0)]), fmt='%0.4f', delimiter=",")            
        np.savetxt(os.path.join(opt.save_path, 'tnr_fold.csv'), np.array([fold_tnr.mean(0), fold_tnr.std(0)]), fmt='%0.4f', delimiter=",")            
        np.savetxt(os.path.join(opt.save_path, 'f1_fold.csv'), np.array([fold_f1.mean(0), fold_f1.std(0)]), fmt='%0.4f', delimiter=",")            
        np.savetxt(os.path.join(opt.save_path, 'acc_fold.csv'), np.array([fold_acc.mean(0), fold_acc.std(0)]), fmt='%0.4f', delimiter=",")            
        np.savetxt(os.path.join(opt.save_path, 'roc_fold.csv'), np.array([fold_roc.mean(0), fold_roc.std(0)]), fmt='%0.4f', delimiter=",")            

            #     if (epoch > 0 and epoch % opt.save_int == 0) or epoch == opt.n_epoch - 1:
            #         torch.save(CNN_model, ('{}/trained_nn.pth'.format(save_folder)))

            # pickle.dump(train_losses, open(os.path.join(save_folder, 'train_losses.p'), 'wb'))
            # pickle.dump(train_acc, open(os.path.join(save_folder, 'train_acc.p'), 'wb'))

        # pdb.set_trace()




