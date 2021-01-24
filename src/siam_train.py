import sys
import os
import time
import json
import random
import math
import numpy as np
import argparse
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.loss import *
from models.single_track import *
from dataset.siam_dataset import *
from utils.config import opt as options
from utils.utils import *
from utils.scheduler import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main(**kwargs):

    options._parse(kwargs)        
    args = options
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch.cuda.manual_seed(317)

    with open(args.single_track, 'r') as outfile:
        args.single_track_data = json.load(outfile)

    model = SiamRPNPP()
    optimizer, lr_scheduler = build_opt_lr(model, args.single_track_start_epoch, args)
    if args.single_track_load_model:
        model, optimizer = load_model(model, opt.single_track_load_model, optimizer)

    # model = model.cuda()
    model = model.eval()
    
    for epoch in range(args.single_track_start_epoch, args.single_track_num_epochs):
        if args.backbone_train_epoch == epoch:
            optimizer, lr_scheduler = build_opt_lr(model, epoch, args)

        lr_scheduler.step(epoch)
        cur_lr = lr_scheduler.get_cur_lr()

        train(model, optimizer, epoch, args)

        if epoch + 1 % 100 == 0:
            path = os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch))
            save_model(path, epoch, model, optimizer)


def train(model, optimizer, epoch, args):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        SiameseDataset(args.single_track_data, train=True),
        batch_size=args.single_track_batch_size)

    model.train()
    end = time.time()

    for i, (z, x, regression_target, conf_target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # z = z.cuda()
        z = Variable(z)
        # x = x.cuda()
        x = Variable(x)

        pred_score, pred_regression = model(z, x)
        b, a2, h, w = pred_score.size()

        pred_conf = pred_score.reshape(-1, 2, 5 * h * w).permute(0, 2, 1)

        pred_offset = pred_regression.reshape(-1, 4, 5 * h * w).permute(0, 2, 1)

        regression_target = regression_target.type(torch.FloatTensor) #.cuda()
        conf_target = conf_target.type(torch.LongTensor)#.cuda()


        cls_loss = rpn_cross_entropy(pred_conf, conf_target)
        reg_loss = rpn_smoothL1(pred_offset, regression_target, conf_target)

        loss = cls_loss + reg_loss

        losses.update(loss.item(), x.size(0))
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i+1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def build_lr_scheduler(optimizer, args, epochs=50, last_epoch=-1):
    warmup_epoch = args.single_track_lr_warm_epoch
    sc1 = StepScheduler(optimizer, last_epoch=last_epoch, epochs=warmup_epoch, new_allowed=True)
    sc2 = LogScheduler(optimizer, last_epoch=last_epoch, epochs=epochs - warmup_epoch, new_allowed=True)
    return WarmUPScheduler(optimizer, sc1, sc2, epochs, last_epoch)


def build_opt_lr(model, current_epoch=0, args=None):
    if current_epoch >= 20:
        for layer in ['layer2', 'layer3', 'layer4']:
            for param in getattr(model.features, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.features, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    else:
        for param in model.features.parameters():
            param.requires_grad = False
        for m in model.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad, model.features.parameters()), 'lr': 0.1 * args.original_lr}]
    trainable_params += [{'params': model.neck.parameters(), 'lr': args.original_lr}]
    trainable_params += [{'params': model.head.parameters(), 'lr': args.original_lr}]

    optimizer = torch.optim.SGD(trainable_params, momentum=args.momentum, weight_decay=args.decay)

    lr_scheduler = build_lr_scheduler(optimizer, args, epochs=args.single_track_num_epochs)
    lr_scheduler.step(args.single_track_start_epoch)
    return optimizer, lr_scheduler


if __name__ == '__main__':
    main()        