#######################################################################################################################
#
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2017, Soumith Chintala. All rights reserved.
# ********************************************************************************************************************
#
#
# The code in this file is adapted from: https://github.com/pytorch/examples/tree/master/imagenet/main.py
#
# Main Difference from the original file: add the networks using partial convolution based padding
#
# Network options using zero padding:               vgg16_bn, vgg19_bn, resnet50, resnet101, resnet152, ... 
# Network options using partial conv based padding: pdvgg16_bn, pdvgg19_bn, pdresnet50, pdresnet101, pdresnet152, ...
#
# Contact: Guilin Liu (guilinl@nvidia.com)
#
#######################################################################################################################
import argparse
import os
import random
import shutil
import time
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import torchvision.models as models_baseline # networks with zero padding
import models as models_partial # partial conv based padding
from utils.data_transform import data_transforms
from utils.visualize_augs import visualize_augmentations, dataset_loop
import matplotlib.pyplot as plt
from utils.dataset import AlbumentationsDataset
from utils.data_frame_helper import read_df
from torch.utils.tensorboard import SummaryWriter
from infer import inference
from utils.metrics import calculate_metrics, attach_to_tensorboard



model_baseline_names = sorted(name for name in models_baseline.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models_baseline.__dict__[name]))

model_partial_names = sorted(name for name in models_partial.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models_partial.__dict__[name]))

model_names = model_baseline_names + model_partial_names


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--data_train', metavar='DIRTRAIN',
                    help='path to training dataset')

parser.add_argument('--data_val', metavar='DIRVAL',
                    help='path to validation dataset')                    

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
# parser.add_argument('--epochs', default=90, type=int, metavar='N',
#                     help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=256, type=int,
#                     metavar='N', help='mini-batch size (default: 256)')
# use the batch size 256 or 192 depending on the memeory
parser.add_argument('-b', '--batch-size', default=192, type=int,
                    metavar='N', help='mini-batch size (default: 192)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--chkpnt_path', default=None, type=str,
                    help='path to checkpoint if pretrained is True')

parser.add_argument('--prefix', default='', type=str)
parser.add_argument('--ckptdirprefix', default='', type=str)

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    root = r'/home/projects/yonina/SAMPL_training/covid_partialconv'
    log_dir_saving_path = os.path.join(root, args.ckptdirprefix, args.prefix)
    writer = SummaryWriter(log_dir=log_dir_saving_path)
    checkpoint_dir = args.ckptdirprefix + 'checkpoint_' + args.arch + '_' + args.prefix + '/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    args.logger_fname = os.path.join(checkpoint_dir, 'loss.txt')

    with open(args.logger_fname, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)    
        log_file.write('world size: %d\n' % args.world_size)
		
		
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch in models_baseline.__dict__:
            model = models_baseline.__dict__[args.arch](pretrained=True)
        else:
            model = models_partial.__dict__[args.arch](pretrained=True)
        # load checkpoint weights if path is given
        if args.chkpnt_path is not None:
            print("=> loading chekpoint '{}'".format(args.chkpnt_path))
            checkpoint = torch.load(args.chkpnt_path)
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
        # model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch in models_baseline.__dict__:
            model = models_baseline.__dict__[args.arch]()
        else:
            model = models_partial.__dict__[args.arch]()
        # model = models.__dict__[args.arch]()


    # logging
    with open(args.logger_fname, "a") as log_file:
        log_file.write('model created\n')
		
		
    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DataParallel(model)
    else:
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        if args.arch.startswith('alexnet') or 'vgg' in args.arch:
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']

            model.load_state_dict(checkpoint['state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            assert False

    cudnn.benchmark = False


    # txt file to data frame
    traindir = args.data_train #os.path.join(args.data, 'train')
    valdir = args.data_val  #os.path.join(args.data, 'val')
    root = os.path.split(os.path.normpath(traindir))[0]
    checkpoint_dir = os.path.join(root, checkpoint_dir)
    train_image_mask = pd.read_csv(os.path.join(traindir, 'paths.txt'), sep=' ', header=None)
    train_image_mask.columns = ['train_image', 'train_mask', 'train_cls']
    test_image_mask = pd.read_csv(os.path.join(valdir, 'paths.txt'), sep=' ', header=None)
    test_image_mask.columns = ['test_image', 'test_mask', 'test_cls']


    # Data loading code
    train_dataset = AlbumentationsDataset(train_image_mask['train_image'], train_image_mask['train_mask'], train_image_mask['train_cls'], phase='TRAIN', transform=data_transforms('TRAIN'))
    val_dataset = AlbumentationsDataset(test_image_mask['test_image'], test_image_mask['test_mask'], test_image_mask['test_cls'], phase='VAL', transform=data_transforms('VAL'))

    # visualize_augmentations(val_dataset, val_dataset.transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # visualize_augmentations(train_dataset, albu_transforms)


    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # logging
    with open(args.logger_fname, "a") as log_file:
        log_file.write('training/val dataset created\n')


    if args.evaluate:
        inference(val_loader, model, criterion, args, root)
        return


    # logging
    with open(args.logger_fname, "a") as log_file:
        log_file.write('started training\n')


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, writer, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': args.arch,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer' : optimizer.state_dict(),
        # }, is_best, foldername=checkpoint_dir, filename='checkpoint.pth')


        if epoch >= 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, False, foldername=log_dir_saving_path, filename='epoch_'+str(epoch)+'_checkpoint.pth')


def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, mask, img_names) in enumerate(train_loader, 0):
        # Clear gradients
        optimizer.zero_grad()
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if len(input.shape) == 3:
            input = input.unsqueeze(1)

        input = input.permute(0, 3, 1, 2)
        mask = mask.unsqueeze(1).type(torch.float)
        # b, c, h, w = mask.shape
        # mask = mask.expand(b, 3, h, w)
        output, fc_embeddings = model(input, mask)
        # output = model(input, mask)
        loss = criterion(output, target)
        metrics = calculate_metrics(output.to('cpu'), target.to('cpu'))
        writer.add_scalar("Loss/train", loss, epoch)
        attach_to_tensorboard(writer, 'train', metrics, epoch)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

            with open(args.logger_fname, "a") as log_file:
                log_file.write('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, writer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, mask, img_names) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            input = input.permute(0, 3, 1, 2)
            output, fc_embeddings = model(input, mask)
            loss = criterion(output, target)

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            metrics = calculate_metrics(output.to('cpu'), target.to('cpu'))
            writer.add_scalar("Loss/train", loss, epoch)
            attach_to_tensorboard(writer, 'val', metrics, epoch)
            # top1.update(prec1[0], input.size(0))
            # top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

                with open(args.logger_fname, "a") as log_file:
                    log_file.write('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        with open(args.logger_fname, "a") as final_log_file:
            final_log_file.write(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, foldername = '', filename='checkpoint.pth'):
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    torch.save(state, os.path.join(foldername, filename))
    if is_best:
        shutil.copyfile(os.path.join(foldername, filename), os.path.join(foldername, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
