import argparse
import logging
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import time

import numpy as np
import torch
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import wandb
from tqdm import tqdm
import pdb
from nt_xent import NT_Xent
import sys
from sklearn.metrics import confusion_matrix, roc_auc_score
from vgg19_64 import Vgg19, Vgg19_multiscale
from CustomDataset import CustomDataset, CustomDatasetTest
from MultiscaleDataset import MultiscaleDataset, MultiscaleDatasetTest
from ImbalancedSampler import ImbalancedDatasetSampler, DistributedSamplerWrapper
from data import DATASET_GETTERS
from models import WideResNet, ModelEMA
from utils import (AverageMeter, accuracy, create_loss_fn,
                   save_checkpoint, reduce_tensor, model_load_state_dict, metrics)
from torchmetrics.classification import MulticlassAUROC
from torchvision.transforms.functional import center_crop
from torch.nn import DataParallel
import glob
import torchvision
# import cls_hrnet as HRNet
import cls_hrnet_ms as HRNet

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--data_path', default="/media/adam/dc156fa0-1275-46c2-962c-bc8c9fcf1cb0/ucr_data/data1/contrastive_learning/dataset/SimCLR_datasets/four_class_hetero", type=str, help='data path')
parser.add_argument('--save_path', default="/media/adam/dc156fa0-1275-46c2-962c-bc8c9fcf1cb0/ucr_data/data1/contrastive_learning/MPL_save/balanced_MPL_ML_MS_HRNet", type=str, help='save path')
parser.add_argument('--dataset', default='HRNet_multiscale', type=str,
                    choices=['custom', 'cifar10', 'cifar100', 'multiscale', 'HRNet', 'HRNet_multiscale'], help='dataset name')
parser.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')
parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
parser.add_argument('--total-steps', default=300000, type=int, help='number of total steps to run')
parser.add_argument('--train_epochs', default=200, type=int, help='number of epochs to run')
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--pretrain_epochs', default=200, type=int)
parser.add_argument('--pretrain_lr', default=5e-3, type=float)
parser.add_argument('--eval-step', default=10, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--num-classes', default=4, type=int, help='number of classes')
parser.add_argument('--resize', default=224, type=int, help='resize image')
parser.add_argument('--batch-size', default=16, type=int, help='train batch size')
parser.add_argument('--projection_dim', default=128, type=int)
parser.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--teacher_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--student_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
parser.add_argument('--weight-decay', default=0, type=float, help='train weight decay')
parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--grad-clip', default=1e9, type=float, help='gradient norm clipping')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
parser.add_argument('--evaluate', default=False, action='store_true', help='only evaluate model on validation set')
parser.add_argument('--finetune', default=False, action='store_true', help='only finetune model on labeled dataset')
parser.add_argument('--finetune-epochs', default=200, type=int, help='finetune epochs')
parser.add_argument('--finetune-batch-size', default=64, type=int, help='finetune batch size')
parser.add_argument('--finetune-lr', default=3e-5, type=float, help='finetune learning late')
parser.add_argument('--finetune-weight-decay', default=0, type=float, help='finetune weight decay')
parser.add_argument('--finetune-momentum', default=0.9, type=float, help='finetune SGD Momentum')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.5, type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--ntxent_temp', default=0.5)
parser.add_argument('--lambda-u', default=8, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda-steps', default=5000, type=float, help='warmup steps of lambda-u')
parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument("--nfolds", default=5)
parser.add_argument("--resample", action='store_true')
parser.add_argument("--n_scales", default=2)
parser.add_argument("--eval_teacher", action='store_true')
parser.add_argument("--optim", default='Adam')
parser.add_argument("--weighted", default=False)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def pretrain(args, labeled_loader, test_loader, teacher_model, criterion, t_optimizer,
             t_scheduler, t_scaler):

    # for epoch in range(args.pretrain_epochs):
    pretrain_crit = torch.nn.CrossEntropyLoss()
    pretrain_optim = torch.optim.SGD(teacher_model.parameters(), lr=5e-3, momentum=0.9, weight_decay=0.0001)
    for epoch in range(args.pretrain_epochs):

        if epoch > 0 and epoch % 100 == 0:
            
            pretrain_optim = torch.optim.SGD(teacher_model.parameters(), lr=5e-4, momentum=0.9, weight_decay=0.0001)

        teacher_model.train()
        loss_value = 0
        for step, ((x_i, x_j, img, _, target, _)) in enumerate(labeled_loader):

            # img1, img2 = img
            # x_i = x_i.to(device1)
            # x_j = x_j.to(device1)
            # img = img.to(device1)
            # target = target.to(device1)

            sys.stdout.write('\rPre-train Train Epoch: {}, Batch: {}/{}'.format(epoch+1, step+1, len(labeled_loader)))

            # with amp.autocast(enabled=args.amp):
            if args.dataset in ['multiscale', 'HRNet_multiscale']:
                img_big, img_small = img.chunk(2,1)
                img_small = center_crop(img_small, img_small.shape[-1]//2)
                pre_out, _ = teacher_model(img_big.cuda(), img_small.cuda())
                if epoch == 0 and step == 0:
                    save_image(img_big[:25], os.path.join(args.save_path, 'pretrain_input_big.png'), nrow=5, normalize=True)
                    save_image(img_big[:25], os.path.join(args.save_path, 'pretrain_input_big.png'), nrow=5, normalize=True)

            # # TODO: pad smaller image 
            else:
                pre_out, _ = teacher_model.forward(img.cuda()) # to(device0))  #, img2.to(device0))
            t_loss = pretrain_crit(pre_out, target.cuda())  # to(device0))

            pretrain_optim.zero_grad()
            t_loss.backward()
            pretrain_optim.step()
            # t_scaler.scale(t_loss).backward()
            # if args.grad_clip > 0:
            #     t_scaler.unscale_(t_optimizer)
            #     nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
            # t_scaler.step(t_optimizer)
            # t_scaler.update()
            # t_scheduler.step()
            loss_value += t_loss.item()

        del img, target, pre_out, t_loss
        print('\nXent Loss: {}'.format(loss_value/len(labeled_loader)))
        teacher_model.eval()
        valid = 0
        count = 0
        targets = []
        predictions = []
        for step, ((img, target)) in enumerate(test_loader):

            # x_i = x_i.to(device1)
            # x_j = x_j.to(device1)
            # img = img.to(device1)
            # target = target.to(device1)

            sys.stdout.write('\rPre-train Valid Epoch: {}, Batch: {}/{}'. format(epoch+1, step+1, len(test_loader)))
            if args.dataset in ['multiscale', 'HRNet_multiscale']:
                img_big, img_small = img.chunk(2,1)
                img_small = center_crop(img_small, img_small.shape[-1]//2)
                valid_out, _ = teacher_model(img_big.cuda(), img_small.cuda())
            else:
                valid_out, _ = teacher_model.forward(img.cuda())  # to(device0))

            pred = valid_out.argmax(1)
            valid += pred.eq(target.cuda()).sum()  # .to(device0)
            count += len(pred)
            targets.extend(target)
            predictions.extend(pred.cpu().numpy())

        del img, target, pred

        if epoch == 0 or epoch % 10 == 0:
            conf_mat = confusion_matrix(targets, predictions)
            print(conf_mat)
        print('\nValid Acc: {}'.format(valid/count))
        # print(conf_mat)
    conf_mat = confusion_matrix(targets, predictions)
    pretrain_metrics = metrics(args, conf_mat)
    np.savetxt(os.path.join(args.save_path, "pretrain_metrics.csv"), pretrain_metrics, fmt='%.4f', delimiter=',')
    torch.save(teacher_model.state_dict(), os.path.join(args.save_path, 'pretrained_teacher.pth.tar')) # , _use_new_zipfile_serialization=False)
    return


class PseudoLabelResampler(torch.utils.data.sampler.Sampler):

    def __init__(self, indices, weights):

        self.indices = indices
        self.num_samples = len(self.indices)
        self.weights = weights

    def __iter__(self):

        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):

        return self.num_samples 


def resample(args, t_model, unlabeled_dataset, weighted):

    imgs = []
    pseudo_labels = []
    real_targets = []
    crop_indices = []
    class_weights = np.zeros(args.num_classes)
    unsampled_loader = DataLoader(
        unlabeled_dataset,
        # sampler=(unlabeled_dataset),
        batch_size=args.batch_size, #  * args.mu,
        num_workers=args.workers,
        drop_last=False)
    img_classes = ['Debris', 'Dense', 'Diff', 'Spread']
    t_model.eval()
    pic =0
    for batch, ((x_i, x_j, x, img_idx, target, crop_params)) in enumerate(unsampled_loader):    
        batch_params = np.zeros((len(target), 4))
        sys.stdout.write('\rGathering Pseudo Labels: {}/{}'. format(batch+1, len(unsampled_loader)))

        if args.dataset in ['multiscale', 'HRNet_multiscale']:
            x_big, x_small = x.chunk(2,1)
            x_small = center_crop(x_small, x_small.shape[-1]//2)
            # save_image(x_big[:25], os.path.join(args.save_path, 'big_images_resample.jpg'), normalize=True)
            # save_image(x_small[:25], os.path.join(args.save_path, 'small_images_resample.jpg'), normalize=True)
            out, _ = t_model(x_big.cuda(), x_small.cuda())
        else:
            out, _ = t_model(x.cuda())  # to(device0))

        pred = out.argmax(axis=1)

        # count number of predicted images per class        
        for img_class in range(args.num_classes):
            class_weights[img_class] += torch.sum(pred.eq(img_class)).item()

        pseudo_labels.extend(pred.cpu().numpy())
        if args.eval_teacher:
            for idx, pl in enumerate(pred):
                if pl != target[idx].item():
                    images_big, images_small = x[idx].chunk(2)
                    save_image(images_big, os.path.join(args.save_path, f'pred_{img_classes[pl]}_tar_{img_classes[target[idx].item()]}_{pic}.jpg'), normalize=True)
                    pic+=1
        real_targets.extend(target.numpy())
        imgs.extend(img_idx.numpy())
        for coordinate in range(4):
            batch_params[:, coordinate] = crop_params[coordinate]
        crop_indices.extend(batch_params)

    real_targets, new_pseudo_labels = zip(*((real, preds) for real, preds in zip(real_targets, pseudo_labels) if real not in [4, 5]))
    conf_mat = confusion_matrix(real_targets, new_pseudo_labels)
    out_metrics = metrics(args, conf_mat)
    # TODO: open file and add class weights 
    print(out_metrics)

    # calculate individual image class weights for new dataset 
    imgs, pseudo_labels = zip(*sorted(zip(imgs, pseudo_labels)))
    weights = [1.0 / class_weights[pseudo_labels[idx]] for idx in list(range(len(imgs)))]
    weights = torch.DoubleTensor(weights)

    # perform iterative multinomial sampling
    pseudo_sampler = PseudoLabelResampler(imgs, weights)

    # TODO: build custom loader that uses same image patch samples as input
    if args.dataset in ['multiscale', 'HRNet_multiscale']: 
        unlabeled_dataset = MultiscaleDataset(os.path.join(args.data_path, 'train'), crop_size=args.resize, thresh=0.25, n_scales=args.n_scales, remove_background=False, crop_params=np.array(crop_indices))
    else:
        unlabeled_dataset = CustomDataset(os.path.join(args.data_path, 'train'), crop_size=args.resize, thresh=0.25, remove_background=False, crop_params=np.array(crop_indices))

    if not weighted:

        resampled_loader = DataLoader(
            unlabeled_dataset,
            sampler=pseudo_sampler,
            batch_size=args.batch_size,  #  * args.mu,
            num_workers=args.workers,
            drop_last=True)

    elif weighted:

        resampled_loader = DataLoader(
            unlabeled_dataset,
            # sampler=pseudo_sampler,
            batch_size=args.batch_size,  #  * args.mu,
            num_workers=args.workers,
            drop_last=True)

        class_weights = [1-weight/class_weights.sum() for weight in class_weights]

    return resampled_loader, class_weights  # , kl_loss


def simclrloop(args, student_model, unlabeled_loader, s_optimizer, s_scaler, s_scheduler, ntxent_criterion):

    s_loss_epoch = 0
    student_model.train()
    for batch, ((x_i, x_j, x, img_idx, target)) in enumerate(unlabeled_loader):

        sys.stdout.write('\rSimCLR Iter {}/{}'.format(batch+1, len(unlabeled_loader)))

        s_images = torch.cat((x_i, x_j))

        with amp.autocast(enabled=args.amp):

            _, s_features = student_model(s_images.to(device1))
            z_i = s_features[:args.batch_size]
            z_j = s_features[args. batch_size:] # .chunk(2)
            del s_features
            s_loss = ntxent_criterion(z_i, z_j)

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()
        s_loss_epoch += s_loss.item()

    return s_loss.item()/len(unlabeled_loader)


def train_loop(args, labeled_loader, unlabeled_loader, unlabeled_dataset, test_loader, finetune_dataset,
               teacher_model, student_model, avg_student_model, criterion, ntxent_criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler, fold, class_weights=None):

    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}@{len(labeled_loader)}")
    logger.info(f"   Total steps = {args.total_steps}")

    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    # weighted_criterion_s = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device1))  #.cuda())
    # weighted_criterion_t = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device0))  #.cuda())

    # for author's code formula
    # moving_dot_product = torch.empty(1).to(args.device)
    # limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
    # nn.init.uniform_(moving_dot_product, -limit, limit)

    # for step in range(args.start_step, args.total_steps):
    # MPL train epoch --> Resample --> SimCLR epoch
    epoch = 0
    step = 0
    t_loss_epoch = 0
    s_loss_epoch = 0
    t_losses_csv = []
    s_losses_csv = []
    s_acc = []
    first_step = True
    while epoch < args.train_epochs:

        if epoch % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()
            mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()

        try:
            step+=1
            # labeled image batch
            x_i_t, x_j_t, images_l, idx_t, targets, _ = next(labeled_iter)  # images_l, targets
        except:
            step+=1
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            # images_l, targets = next(labeled_iter)
            x_i_t, x_j_t, images_l, idx_t, targets, _ = next(labeled_iter)  # images_l, targets


        try:
            new_iter = False
            # unlabeled contrastive iteration
            images_uw, images_us, _, _, _, _ = next(unlabeled_iter)
            # (images_uw, images_us), _ = next(unlabeled_iter)
        except:  # one iteration through entire unlabeled dataset == one epoch
            new_iter = True
            if epoch > 0:
                t_losses_csv.append(t_loss_epoch/len(unlabeled_loader))
                s_losses_csv.append(s_loss_epoch/len(unlabeled_loader))

            epoch += 1
            t_loss_epoch = 0
            s_loss_epoch = 0

            if args.resample:
                """resample new pseudo-label dataset"""
                unlabeled_loader, class_weights = resample(args, teacher_model, unlabeled_dataset, weighted=args.weighted)
                print('Epoch {}, Class Weights: {}'.format(epoch+1, class_weights))

            # weighted_criterion_s = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device1)) # .to(device))
            # weighted_criterion_t = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device0)) # .to(device))

            # train SimCLR model with resampled datast
            # sim_loss = simclrloop(args, student_model, unlabeled_loader, s_optimizer,
            #                       s_scaler, s_scheduler, ntxent_criterion)

            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_loader)
            # images_uw ==> original image, images_us ==> transformed image 
            images_uw, images_us, _, idx_u, _, _ = next(unlabeled_iter)
            teacher_model.train()

        # save snapshot of input images
        if first_step == True:
            first_step = False
            save_image(images_uw[:25], os.path.join(args.save_path, 'mpl_input.png'), nrow=5, normalize=True)

        data_time.update(time.time() - end)

        # images_l = images_l.to()  # args.device)
        # images_uw = images_uw.to()  # args.device)
        # images_us = images_us.to()  # args.device)
        # targets = targets.to()  # args.device)
        # teacher --> device0
        # student --> device1
        with amp.autocast(enabled=args.amp):

            # for teacher, concatenate labeled and unlabeled images 
            batch_size = images_l.shape[0]
            # t_images = torch.cat((x_t, x_i, x_j))

            """teacher forward pass"""
            if args.dataset in ['multiscale', 'HRNet_multiscale']:
                img_big_l, img_small_l = images_l.chunk(2,1)
                img_big_uw, img_small_uw = images_uw.chunk(2,1)
                img_big_us, img_small_us = images_us.chunk(2,1)

                img_small_l = center_crop(img_small_l, img_small_l.shape[-1]//2)
                img_small_uw = center_crop(img_small_uw, img_small_uw.shape[-1]//2)
                img_small_us = center_crop(img_small_us, img_small_us.shape[-1]//2)

                images_big = torch.cat((img_big_l, img_big_uw, img_big_us))
                images_small = torch.cat((img_small_l, img_small_uw, img_small_us))
                t_logits, _ = teacher_model(images_big.cuda(), images_small.cuda())

                # t_logits, _ = teacher_modelz(t_images.cuda()) # .to(device0))
                t_logits_l = t_logits[:batch_size]
                t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
                del t_logits

            else:
                t_images = torch.cat((images_l, images_uw, images_us))
                t_logits, _ = teacher_model(t_images.cuda()) # .to(device0))
                t_logits_l = t_logits[:batch_size]
                t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
                del t_logits

            # cross entropy criterion from labeled image logits 
            t_loss_l = criterion(t_logits_l, targets.cuda())  # to(device0))

            """teacher soft pseudo-labels """
            # gather soft labels from unlabeled logits
            soft_pseudo_label = torch.softmax(t_logits_uw.detach() / args.temperature, dim=-1)
            # get tuple of hard pseudo labels (values, indices) for each soft pseudo label
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            # mask ==> boolean of probabilities greather than; equal to a certain threshold (0.95)
            mask = max_probs.ge(args.threshold).float()
            # loss function for soft labels
            # softmax pseudo-labels for normal images x softmax pseudo-labels for transformed images x threshold mask
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
            )
            # how to weight unlabeled data for teacher...
            weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
            # total teacher loss == labeled loss value plus weighted unlabeled loss value 
            # UDA --> unlabeled data augmentation
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            """train student"""
            if args.dataset in ['multiscale', 'HRNet_multiscale']:

                s_images_big = torch.cat((img_big_uw, img_big_us))
                s_images_small = torch.cat((img_small_uw, img_small_us))

                s_images_mpl_big = torch.cat((img_big_l, img_big_us))
                s_images_mpl_small = torch.cat((img_small_l, img_small_us))

                _, s_features = student_model(s_images_big.cuda(), s_images_small.cuda())
                s_logits, _ = student_model(s_images_mpl_big.cuda(), s_images_mpl_small.cuda())

                z_i = s_features[:args.batch_size]
                z_j = s_features[args.batch_size:]

                s_logits_l = s_logits[:args.batch_size]
                s_logits_us = s_logits[args.batch_size:]

                del s_logits, s_features

            else:
                s_images = torch.cat((images_uw, images_us))
                s_images_mpl = torch.cat((images_l, images_us))

                _, s_features = student_model(s_images.cuda()) # to(device1))
                s_logits, _ = student_model(s_images_mpl.cuda()) # to(device1))

                z_i = s_features[:args.batch_size] # s_features_uw
                z_j = s_features[args.batch_size:] # s_features_us

                s_logits_l = s_logits[:batch_size] # s_logits_l
                s_logits_us = s_logits[batch_size:] # s_logits_us

                del s_logits, s_features

            # calculate X-ent loss for labeled student images 
            # s_loss_l_old ==> before student update
            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets.cuda()) # to(device1))
            # calculate x-ent loss for unlabeled images with hard psuedo labels
            s_loss_mpl = criterion(s_logits_us, hard_pseudo_label.cuda()) # to(device1))
            s_loss_ntx = ntxent_criterion(z_i, z_j)
            s_loss = s_loss_mpl  # + s_loss_ntx  


        """make student optimizer step"""
        s_loss_epoch += s_loss.item()
        # update student network in relation to x-ent loss against hard_pseudo_labels 
        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()
        if args.ema > 0:
            avg_student_model.update_parameters(student_model)

        with amp.autocast(enabled=args.amp):

            # perform forward pass of updated student network with labeled images
            with torch.no_grad():
                if args.dataset in ['multiscale', 'HRNet_multiscale']:
                    s_logits_l, _ = student_model(img_big_l.cuda(), img_small_l.cuda())
                else:
                    s_logits_l, _ = student_model(images_l.cuda()) # to(device1))
            # calculate cross entropy loss for updated student network
            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets.cuda()) # to(device1))

            # theoretically correct formula (https://github.com/kekmodel/MPL-pytorch/issues/6)
            # dot_product = s_loss_l_old - s_loss_l_new

            # author's code formula
            # take difference between old and new loss values
            dot_product = s_loss_l_new.item() - s_loss_l_old.item()
            # print(f'dot_prod{dot_product}')
            # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
            # dot_product = dot_product - moving_dot_product

            # collect hard pseudo labels for unlabeled datset 
            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
            # calculate meta-pseudo-label loss for teacher using product of student loss and unlabeled teacher loss 
            # only update teacher if student is learning significantly (reason for dot product of old-->new)
            # t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
            t_loss_mpl = dot_product * criterion(t_logits_us, hard_pseudo_label)
            # test
            # t_loss_mpl = torch.tensor(0.).to(args.device)
            # overall t_loss == loss formula for unsupervised data augmentation and meta-pseudo-label loss function 
            t_loss = t_loss_uda + t_loss_mpl

        """make teacher optimizer step"""
        t_loss_epoch += t_loss.item()
        # update teacher network using combination loss from teacher/student 
        """freeze teacher network (uncomment to update)"""
        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()
        teacher_model.zero_grad()

        student_model.zero_grad()

        if args.world_size > 1:
            s_loss = reduce_tensor(s_loss.detach(), args.world_size)
            t_loss = reduce_tensor(t_loss.detach(), args.world_size)
            t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
            t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
            t_loss_mpl = reduce_tensor(t_loss_mpl.detach(), args.world_size)
            mask = reduce_tensor(mask, args.world_size)

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses_mpl.update(t_loss_mpl.item())
        mean_mask.update(mask.mean().item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Fold: {fold}."
            f"Train Iter: {epoch+1}/{args.train_epochs}. "
            f"Steps:{step}"
            f"LR: {get_lr(s_optimizer):.4e}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        pbar.update()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("lr", get_lr(s_optimizer), epoch)
            wandb.log({"lr": get_lr(s_optimizer)})

        # args.num_eval = step // args.eval_step
        # if (step + 1) % args.eval_step == 0:
        args.num_eval = epoch // args.eval_step
        if (epoch + 1) % args.eval_step == 0 and new_iter==True:
            pbar.close()
            if args.local_rank in [-1, 0]:
                args.writer.add_scalar("train/1.s_loss", s_losses.avg, args.num_eval)
                args.writer.add_scalar("train/2.t_loss", t_losses.avg, args.num_eval)
                args.writer.add_scalar("train/3.t_labeled", t_losses_l.avg, args.num_eval)
                args.writer.add_scalar("train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
                args.writer.add_scalar("train/5.t_mpl", t_losses_mpl.avg, args.num_eval)
                args.writer.add_scalar("train/6.mask", mean_mask.avg, args.num_eval)
                wandb.log({"train/1.s_loss": s_losses.avg,
                           "train/2.t_loss": t_losses.avg,
                           "train/3.t_labeled": t_losses_l.avg,
                           "train/4.t_unlabeled": t_losses_u.avg,
                           "train/5.t_mpl": t_losses_mpl.avg,
                           "train/6.mask": mean_mask.avg})

                test_model = avg_student_model if avg_student_model is not None else student_model
                test_loss, top1, top5, _, _, _ = evaluate(args, test_loader, test_model, criterion)
                s_acc.append(top1)

                args.writer.add_scalar("test/loss", test_loss, args.num_eval)
                args.writer.add_scalar("test/acc@1", top1, args.num_eval)
                args.writer.add_scalar("test/acc@5", top5, args.num_eval)
                wandb.log({"test/loss": test_loss,
                           "test/acc@1": top1,
                           "test/acc@5": top5})

                is_best = top1 > args.best_top1
                if is_best:
                    args.best_top1 = top1
                    args.best_top5 = top5

                logger.info(f"top-1 acc: {top1:.2f}")
                logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

                save_checkpoint(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'best_top1': args.best_top1,
                    'best_top5': args.best_top5,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
                }, is_best)
                print('\nCheckpoint Saved!')

                # save t and s loss array
                np.savetxt(os.path.join(args.save_path, 'teacher_loss.csv'), np.array(t_losses_csv), fmt='%.4f', delimiter=',')
                np.savetxt(os.path.join(args.save_path, 'student_loss.csv'), np.array(s_losses_csv), fmt='%.4f', delimiter=',')
                np.savetxt(os.path.join(args.save_path, 'student_acc.csv'), np.array(s_acc), fmt='%0.4f', delimiter=',')

    if args.local_rank in [-1, 0]:
        args.writer.add_scalar("result/test_acc@1", args.best_top1)
        wandb.log({"result/test_acc@1": args.best_top1})

    # reload best student model
    checkpoint = torch.load(os.path.join(args.save_path, '{}_best.pth.tar'.format(args.name)))  # , map_location=loc)
    model_load_state_dict(student_model, checkpoint['student_state_dict'])
    print('reloaded best student model.')
    # finetune and evaluate model
    finetune(args, finetune_dataset, labeled_loader, student_model, criterion, last_epoch=True)
    args.finetune = True
    _, _, _, out_metrics, conf_mat, roc_score = evaluate(args, test_loader, student_model, criterion)

    del t_scaler, t_scheduler, t_optimizer, teacher_model, labeled_loader, unlabeled_loader
    del s_scaler, s_scheduler, s_optimizer
    ckpt_name = f'{args.save_path}/{args.name}_best.pth.tar'
    loc = f'cuda:{args.gpu}'
    checkpoint = torch.load(ckpt_name, map_location=loc)
    logger.info(f"=> loading checkpoint '{ckpt_name}'")
    if checkpoint['avg_state_dict'] is not None:
        model_load_state_dict(student_model, checkpoint['avg_state_dict'])
    else:
        model_load_state_dict(student_model, checkpoint['student_state_dict'])

    return out_metrics, roc_score 


def evaluate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    labels = []
    predictions = []
    probs = []
    roc_auc = MulticlassAUROC(num_classes=args.num_classes, average='none')
    with torch.no_grad():
        end = time.time()
        for step, (images, targets) in enumerate(test_iter):

            sys.stdout.write('\rEval Iter {}/{}'.format(step+1, len(test_iter)))
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            # images = images.cuda() # to(device1)
            # targets = targets.cuda() # to(device1)
            with amp.autocast(enabled=args.amp):
                if args.dataset in ['multiscale', 'HRNet_multiscale']:
                    img_big, img_small = images.chunk(2,1)
                    img_small = center_crop(img_small, img_small.shape[-1]//2)
                    outputs, _ = model(img_big.cuda(), img_small.cuda())
                else:
                    outputs, _ = model(images.cuda())
                loss = criterion(outputs, targets.cuda())

            batch_probs = F.softmax(outputs).cpu().numpy()
            # check if converted values sum to one
            probs.extend(F.softmax(outputs, dim=1).detach().cpu())  # .numpy())
            predictions.extend(outputs.argmax(1).cpu().numpy())
            labels.extend(targets.cpu().numpy())
            acc1, acc5 = accuracy(outputs, targets, (1, 4))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. ")
        conf_mat = confusion_matrix(labels, predictions)
        print(conf_mat)
        out_metrics = metrics(args, conf_mat)
        roc_score = roc_auc(torch.cat(probs).reshape(-1, args.num_classes), torch.tensor(labels))
        print(out_metrics)
        if args.finetune == True:
            np.savetxt(os.path.join(args.save_path, "finetune_metrics.csv"), out_metrics, fmt='%.4f', delimiter=',')
            np.savetxt(os.path.join(args.save_path, "finetune_confmat.csv"), conf_mat, fmt='%i', delimiter=',')
            np.savetxt(os.path.join(args.save_path, "finetune_roc_auc.csv"), roc_score, fmt='%.4f', delimiter=',')
        elif args.eval_teacher == True:
            np.savetxt(os.path.join(args.save_path, "teacher_metrics.csv"), out_metrics, fmt='%.4f', delimiter=',')
            np.savetxt(os.path.join(args.save_path, "teacher_confmat.csv"), conf_mat, fmt='%i', delimiter=',')
            np.savetxt(os.path.join(args.save_path, "teacher_roc_auc.csv"), roc_score, fmt='%.4f', delimiter=',')
        else:
            np.savetxt(os.path.join(args.save_path, "eval_metrics.csv"), out_metrics, fmt='%.4f', delimiter=',')
            np.savetxt(os.path.join(args.save_path, "eval_roc_auc.csv"), roc_score, fmt='%.4f', delimiter=',')


        test_iter.close()
        return losses.avg, top1.avg, top5.avg, out_metrics, conf_mat, roc_score


def finetune(args, finetune_dataset, test_loader, model, criterion, last_epoch=False):

    # test trained contrastive learing model before finetune
    # evaluate(args, test_loader, model, criterion)
    """Load Best model from save_path"""

    model.drop = nn.Identity()
    labeled_loader = test_loader
    # train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    # labeled_loader = DataLoader(
    #     finetune_dataset,
    #     batch_size=args.finetune_batch_size,
    #     num_workers=args.workers,
    #     pin_memory=True)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.finetune_lr,
                          momentum=args.finetune_momentum,
                          weight_decay=args.finetune_weight_decay,
                          nesterov=True)

    scaler = amp.GradScaler(enabled=args.amp)

    logger.info("***** Running Finetuning *****")
    logger.info(f"   Finetuning steps = {len(labeled_loader)*args.finetune_epochs}")

    for epoch in range(args.finetune_epochs):
        if args.world_size > 1:
            labeled_loader.sampler.set_epoch(epoch + 624)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        labeled_iter = tqdm(labeled_loader, disable=args.local_rank not in [-1, 0])
        for step, (_, _, images, _, targets, _) in enumerate(labeled_iter):

            sys.stdout.write('\rFinetune Epoch: {}, Iter: {}/{}'. format(epoch+1, step+1, len(labeled_iter)))
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            # images = images.cuda() # to(device1)
            # targets = targets.cuda() # to(device1)

            with amp.autocast(enabled=args.amp):

                model.zero_grad()
                if args.dataset in ['multiscale', 'HRNet_multiscale']:
                    img_big, img_small = images.chunk(2,1)
                    img_small = center_crop(img_small, img_small.shape[-1]//2)
                    outputs, _ = model(img_big.cuda(), img_small.cuda())
                else:
                    outputs, _ = model(images.cuda())
                loss = criterion(outputs, targets.cuda())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.world_size > 1:
                loss = reduce_tensor(loss.detach(), args.world_size)
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            labeled_iter.set_description(
                f"Finetune Epoch: {epoch+1:2}/{args.finetune_epochs:2}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
        labeled_iter.close()

        """evaluate separately (commented out)"""
        if not args.resume and not last_epoch:  # args.local_rank in [-1, 0]:
            args.writer.add_scalar("finetune/train_loss", losses.avg, epoch)
            test_loss, top1, top5, _, _, _ = evaluate(args, test_loader, model, criterion)
            args.writer.add_scalar("finetune/test_loss", test_loss, epoch)
            args.writer.add_scalar("finetune/acc@1", top1, epoch)
            args.writer.add_scalar("finetune/acc@5", top5, epoch)
            wandb.log({"finetune/train_loss": losses.avg,
                       "finetune/test_loss": test_loss,
                       "finetune/acc@1": top1,
                       "finetune/acc@5": top5})

            is_best = top1 > args.best_top1
            if is_best:
                args.best_top1 = top1
                args.best_top5 = top5

            logger.info(f"top-1 acc: {top1:.2f}")
            logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

            save_checkpoint(args, {
                'step': step + 1,
                'best_top1': args.best_top1,
                'best_top5': args.best_top5,
                'student_state_dict': model.state_dict(),
                'avg_state_dict': None,
                'student_optimizer': optimizer.state_dict(),
            }, is_best, finetune=True)
            print('\nCheckpoint Saved!')
        # if args.local_rank in [-1, 0]:
            args.writer.add_scalar("result/finetune_acc@1", args.best_top1)
            wandb.log({"result/finetune_acc@1": args.best_top1})

    save_checkpoint(args, {
        'step': step + 1,
        'best_top1': args.best_top1,
        'best_top5': args.best_top5,
        'student_state_dict': model.state_dict(),
        'avg_state_dict': None,
        'student_optimizer': optimizer.state_dict(),
    }, is_best=False, finetune=True)
    print('\nCheckpoint Saved!')
    return


def main():

    args = parser.parse_args()

    fold_tpr = []
    fold_tnr = []
    fold_f1 = []
    fold_roc = []
    fold_acc =[]

    if args.resume:
        args.nfolds = 1
        
    for fold in range(args.nfolds):  
      
        args = parser.parse_args()
        base_path = args.save_path
        args.save_path = os.path.join(args.save_path, 'fold_{}'.format(fold))
        args.data_path = os.path.join(args.data_path, 'fold_{}'.format(fold))

        args.best_top1 = 0.
        args.best_top5 = 0.

        if args.local_rank != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            args.gpu = args.local_rank
            torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.world_size)
            args.world_size = torch.distributed.get_world_size()
        else:
            args.gpu = 0
            args.world_size = 1

        device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)

        logger.warning(
            f"Process rank: {args.local_rank}, "
            f"device: {device0, device1}, "
            f"distributed training: {bool(args.local_rank != -1)}, "
            f"16-bits training: {args.amp}")

        logger.info(dict(args._get_kwargs()))

        if args.local_rank in [-1, 0]:
            args.writer = SummaryWriter(f"results/{args.name}")
            wandb.init(mode='disabled', name=args.name, project='MPL', config=args)

        if args.seed is not None:
            set_seed(args)

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        os.makedirs(args.save_path, exist_ok=True)

        if args.dataset in ['custom', 'HRNet']:
            # labeled_dataset, unlabeled_dataset, test_dataset, finetune_dataset = DATASET_GETTERS[args.dataset](args)
            unlabeled_dataset = CustomDataset(os.path.join(args.data_path, 'train'), crop_size=args.resize, thresh=0.25, remove_background=False
                )
            labeled_dataset = CustomDataset(os.path.join(args.data_path, 'test'), crop_size=args.resize, thresh=0.25, remove_background=False
                )
            test_dataset = CustomDatasetTest(os.path.join(args.data_path, 'valid'), crop_size=args.resize, thresh=0.25
                )
            finetune_dataset = labeled_dataset

        elif args.dataset in ['multiscale', 'HRNet_multiscale']:

            unlabeled_dataset = MultiscaleDataset(os.path.join(args.data_path, 'train'), crop_size=args.resize, thresh=0.25, n_scales=args.n_scales, remove_background=False
                )
            labeled_dataset = MultiscaleDataset(os.path.join(args.data_path, 'test'), crop_size=args.resize, thresh=0.25, n_scales=args.n_scales,  remove_background=False
                )
            test_dataset = MultiscaleDatasetTest(os.path.join(args.data_path, 'valid'), crop_size=args.resize, thresh=0.25, n_scales=args.n_scales
                )
            finetune_dataset = labeled_dataset

        if args.local_rank == 0:
            torch.distributed.barrier()

        train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
        imbalanced_sampler = ImbalancedDatasetSampler(labeled_dataset)
        labeled_loader = DataLoader(
            labeled_dataset,
            sampler=imbalanced_sampler,  # DistributedSamplerWrapper(imbalanced_sampler)
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True)

        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            sampler=train_sampler(unlabeled_dataset),
            batch_size=args.batch_size, #  * args.mu,
            # shuffle=True,
            num_workers=args.workers,
            drop_last=True)

        test_loader = DataLoader(test_dataset,
                                 # sampler=SequentialSampler(test_dataset),
                                 batch_size=args.batch_size,
                                 # num_workers=args.workers,
                                 shuffle=False)

        make_example_images = False
        if make_example_images:
            image_save_path = '/media/adam/dc156fa0-1275-46c2-962c-bc8c9fcf1cb0/ucr_data/data1/contrastive_learning/MPL_save/crop_samples'
            data_path = '/media/adam/dc156fa0-1275-46c2-962c-bc8c9fcf1cb0/ucr_data/data1/HDNeuron/GAN_imgs'
            img_classes = ['Debris', 'Dense', 'Diff', 'Spread']
            for image_class in img_classes:
                if args.dataset == 'custom':

                    test_dataset = CustomDatasetTest(os.path.join(data_path, image_class), crop_size=args.resize, thresh=0.25
                        )
                    finetune_dataset = labeled_dataset

                elif args.dataset in ['multiscale', 'HRNet_multiscale']:

                    test_dataset = MultiscaleDatasetTest(os.path.join(data_path, image_class), crop_size=args.resize, thresh=0.25, n_scales=args.n_scales
                        )
                    finetune_dataset = labeled_dataset

                if args.local_rank == 0:
                    torch.distributed.barrier()

                test_loader = DataLoader(test_dataset,
                                         # sampler=SequentialSampler(test_dataset),
                                         batch_size=args.batch_size,
                                         # num_workers=args.workers,
                                         shuffle=True)

                for step, (images, _) in enumerate(test_loader): 
                    if step == 5:
                        break
                    if args.dataset == 'custom':
                        save_image(images[:25], os.path.join(image_save_path, f'128_images_{image_class}_{step}.png'), nrow=5, normalize=True)
                    elif args.dataset in ['multiscale', 'HRNet_multiscale']:
                        images_big, images_small = images.chunk(2,1)
                        images_small = torchvision.transforms.CenterCrop(112)(images_small)
                        save_image(images_big[:25], os.path.join(image_save_path, f'224_images_{image_class}_{step}.png'), nrow=5, normalize=True)
                        save_image(images_small[:25], os.path.join(image_save_path, f'112_images_{image_class}_{step}.png'), nrow=5, normalize=True)


        if args.dataset == "cifar10":
            depth, widen_factor = 28, 2
        elif args.dataset == "custom":
            depth, widen_factor = 28, 2
        elif args.dataset == 'cifar100':
            depth, widen_factor = 28, 8

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        if args.dataset == 'custom':
            teacher_model = Vgg19(args.num_classes, args.projection_dim)
            student_model = Vgg19(args.num_classes, args.projection_dim)
        elif args.dataset == 'multiscale':
            teacher_model = Vgg19_multiscale(args.num_classes, args.projection_dim)
            student_model = Vgg19_multiscale(args.num_classes, args.projection_dim)
        elif args.dataset == 'HRNet':
            teacher_model = HRNet.get_cls_net()
            student_model = HRNet.get_cls_net()
        elif args.dataset == 'HRNet_multiscale':
            teacher_model = HRNet.get_cls_net()
            student_model = HRNet.get_cls_net()

        if args.local_rank == 0:
            torch.distributed.barrier()

        logger.info(f"Model: VGG19 {args.num_classes}x{args.projection_dim}") # # WideResNet
        logger.info(f"Params: {sum(p.numel() for p in teacher_model.parameters())/1e6:.2f}M")

        # teacher_model.to(args.device)
        # student_model.to(args.device)
        avg_student_model = None
        if args.ema > 0:
            avg_student_model = ModelEMA(student_model, args.ema)

        criterion = create_loss_fn(args)  # --> to.(device)? 
        # TODO: add nt_xent criterion for SimCLR training
        ntxent_criterion = NT_Xent(args.batch_size, args.ntxent_temp, 1)


        no_decay = ['bn']
        teacher_parameters = [
            {'params': [p for n, p in teacher_model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in teacher_model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        student_parameters = [
            {'params': [p for n, p in student_model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in student_model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if args.optim == 'SGD': 
            print("USING SGD OPTIMIZER.")
            t_optimizer = optim.SGD(teacher_parameters,
                                    lr=args.teacher_lr,
                                    momentum=args.momentum,
                                    nesterov=args.nesterov)
            s_optimizer = optim.SGD(student_parameters,
                                    lr=args.student_lr,
                                    momentum=args.momentum,
                                    nesterov=args.nesterov)
        else:
            print("USING ADAM OPTIMIZER")
            t_optimizer = optim.Adam(teacher_parameters, lr=5e-3)
            s_optimizer = optim.Adam(student_parameters, lr=5e-3)

        t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                      args.warmup_steps,
                                                      args.total_steps,
                                                      )
        s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                      args.warmup_steps,
                                                      args.total_steps,
                                                      args.student_wait_steps)

        t_scaler = amp.GradScaler(enabled=args.amp)
        s_scaler = amp.GradScaler(enabled=args.amp)

        # optionally resume from a checkpoint
        if args.resume or args.eval_teacher:

            if args.eval_teacher:
                fold_resume = os.path.join(args.save_path, 'custom_best.pth.tar')
            else:
                fold_resume = args.resume
            if os.path.isfile(fold_resume):
                logger.info(f"=> loading checkpoint '{fold_resume}'")
                loc = f'cuda:{args.gpu}'
                checkpoint = torch.load(fold_resume, map_location=loc)
                args.best_top1 = checkpoint['best_top1'].to(torch.device('cpu'))
                args.best_top5 = checkpoint['best_top5'].to(torch.device('cpu'))
                if not (args.evaluate or args.finetune):
                    args.start_step = checkpoint['step']
                    t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                    s_optimizer.load_state_dict(checkpoint['student_optimizer'])
                    t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                    s_scheduler.load_state_dict(checkpoint['student_scheduler'])
                    t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                    s_scaler.load_state_dict(checkpoint['student_scaler'])
                    model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])
                    if avg_student_model is not None:
                        model_load_state_dict(avg_student_model, checkpoint['avg_state_dict'])
                else:
                    if checkpoint['avg_state_dict'] is not None:
                        model_load_state_dict(student_model, checkpoint['avg_state_dict'])
                    else:
                        model_load_state_dict(student_model, checkpoint['student_state_dict'])

                logger.info(f"=> loaded checkpoint '{fold_resume}' (step {checkpoint['step']})")
                completed_epochs = int(np.floor(checkpoint['step']/len(unlabeled_loader)))-1
                args.train_epochs = args.train_epochs - completed_epochs
            else:
                logger.info(f"=> no checkpoint found at '{fold_resume}'")

        if args.local_rank != -1:
            teacher_model = nn.parallel.DistributedDataParallel(
                teacher_model, device_ids=[args.local_rank],
                output_device=args.local_rank, find_unused_parameters=True)
            student_model = nn.parallel.DistributedDataParallel(
                student_model, device_ids=[args.local_rank],
                output_device=args.local_rank, find_unused_parameters=True)

        if args.local_rank == -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            teacher_model = DataParallel(teacher_model).cuda()
            student_model = DataParallel(student_model).cuda()
            # teacher_model = teacher_model.to(device0)
            # student_model = student_model.to(device1)

        """evaluate teacher model"""
        if args.eval_teacher:
            del t_scaler, t_scheduler, t_optimizer, unlabeled_loader
            del s_scaler, s_scheduler, s_optimizer, student_model
            evaluate(args, test_loader, teacher_model, criterion)

            continue

        """finetune student with labeled dataset"""
        if args.finetune:
            del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
            del s_scaler, s_scheduler, s_optimizer
            # evaluate(args, test_loader, student_model, criterion)
            finetune(args, finetune_dataset, labeled_loader, student_model, criterion)
            evaluate(args, test_loader, student_model, criterion)

            return

        """evaluate the trained model"""
        if args.evaluate:
            del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader, labeled_loader
            del s_scaler, s_scheduler, s_optimizer
            evaluate(args, test_loader, student_model, criterion)
            return

        teacher_model.zero_grad()
        student_model.zero_grad()

        """pretrain teacher model"""
        if args.pretrain == True and not args.resume:
            if os.path.exists(os.path.join(args.save_path, 'pretrained_teacher.pth.tar')) and not args.resume:
                print('Loading Pretrained Teacher...')
                teacher_model.load_state_dict(torch.load(os.path.join(args.save_path, 'pretrained_teacher.pth.tar')))
            elif not args.resume:
                pretrain(args, labeled_loader, test_loader, teacher_model,
                         t_optimizer, criterion, t_scheduler, t_scaler)  #  

        if args.resample == True:
            unlabeled_loader, class_weights = resample(args, teacher_model, unlabeled_dataset, weighted=args.weighted)
            print('Initial Class Weights - Debris:{}, Dense: {}, Diff: {}, Spread: {}'.format(
                class_weights[0], class_weights[1], class_weights[2], class_weights[3]))

            """perform training iterations"""
            out_metrics, roc_score = train_loop(args, labeled_loader, unlabeled_loader, unlabeled_dataset, test_loader, finetune_dataset,
                                            teacher_model, student_model, avg_student_model, criterion, ntxent_criterion,
                                            t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler, fold, class_weights)
        else:
            """perform training iterations"""
            out_metrics, roc_score = train_loop(args, labeled_loader, unlabeled_loader, unlabeled_dataset, test_loader, finetune_dataset,
                                                teacher_model, student_model, avg_student_model, criterion, ntxent_criterion,
                                                t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler, fold)

    return


if __name__ == '__main__':
    
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    main()

    print('\nEnd Training.')
