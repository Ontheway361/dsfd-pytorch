#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from dataset.config import cfg
from layers.modules import MultiBoxLoss
from dataset.widerface import WiderFace, detection_collate
from models.factory import build_net, basenet_factory

parser = argparse.ArgumentParser(description='DSFD face Detector Training With Pytorch')

parser.add_argument('--model',   type=str,    default='vgg', choices=['vgg', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--resume',  type=str,    default='')
parser.add_argument('--workers', type=int,    default=4)    # TODO
parser.add_argument('--cuda',    type=bool,   default=True)
parser.add_argument('--batch_size',type=int,  default=16)
parser.add_argument('--lr',      type=float,  default=1e-3)
parser.add_argument('--momentum',type=float,  default=0.9)
parser.add_argument('--wd',      type=float,  default=5e-4)
parser.add_argument('--gamma',   type=float,  default=0.1)
parser.add_argument('--multigpu',type=bool,   default=False)
parser.add_argument('--save_to', type=str,    default='weights/')
args = parser.parse_args()

if not args.multigpu:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


save_to = os.path.join(args.save_to, args.model)
if not os.path.exists(save_to):
    os.mkdir(save_to)


train_dataset = WiderFace(cfg.FACE.TRAIN_FILE, mode='train')

val_dataset = WiderFace(cfg.FACE.VAL_FILE, mode='val')

train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=True)
val_batchsize = args.batch_size // 2
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)


min_loss = np.inf


def train():
    
    per_epoch_size = len(train_dataset) // args.batch_size
    start_epoch = 0
    iteration = 0
    step_index = 0

    basenet = basenet_factory(args.model)
    dsfd_net = build_net('train', cfg.NUM_CLASSES, args.model)
    net = dsfd_net

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        start_epoch = net.load_weights(args.resume)
        iteration = start_epoch * per_epoch_size
    else:
        base_weights = torch.load(args.save_to + basenet)
        print('Load base network {}'.format(args.save_to + basenet))
        if args.model == 'vgg':
            net.vgg.load_state_dict(base_weights)
        else:
            net.resnet.load_state_dict(base_weights)

    if args.cuda:
        if args.multigpu:
            net = torch.nn.DataParallel(dsfd_net)
        net = net.cuda()
        cudnn.benckmark = True

    if not args.resume:
        print('Initializing weights...')
        dsfd_net.extras.apply(dsfd_net.weights_init)
        dsfd_net.fpn_topdown.apply(dsfd_net.weights_init)
        dsfd_net.fpn_latlayer.apply(dsfd_net.weights_init)
        dsfd_net.fpn_fem.apply(dsfd_net.weights_init)
        dsfd_net.loc_pal1.apply(dsfd_net.weights_init)
        dsfd_net.conf_pal1.apply(dsfd_net.weights_init)
        dsfd_net.loc_pal2.apply(dsfd_net.weights_init)
        dsfd_net.conf_pal2.apply(dsfd_net.weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg, args.cuda)
    print('Loading wider dataset...')
    print('Using the specified args:')
    print(args)

    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

    net.train()
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True)
                           for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)

            loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2
            loss.backward()
            optimizer.step()
            t1 = time.time()
            losses += loss.data[0]

            if iteration % 10 == 0:
                tloss = losses / (batch_idx + 1)
                print('Timer: %.4f' % (t1 - t0))
                print('epoch:' + repr(epoch) + ' || iter:' +
                      repr(iteration) + ' || Loss:%.4f' % (tloss))
                print('->> pal1 conf loss:{:.4f} || pal1 loc loss:{:.4f}'.format(
                    loss_c_pal1.data[0], loss_l_pa1l.data[0]))
                print('->> pal2 conf loss:{:.4f} || pal2 loc loss:{:.4f}'.format(
                    loss_c_pal2.data[0], loss_l_pa12.data[0]))
                print('->>lr:{}'.format(optimizer.param_groups[0]['lr']))

            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                file = 'dsfd_' + repr(iteration) + '.pth'
                torch.save(dsfd_net.state_dict(),
                           os.path.join(save_to, file))
            iteration += 1

        val(epoch, net, dsfd_net, criterion)
        if iteration == cfg.MAX_STEPS:
            break


def val(epoch, net, dsfd_net, criterion):
    net.eval()
    step = 0
    losses = 0
    t1 = time.time()
    for batch_idx, (images, targets) in enumerate(val_loader):
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True)
                       for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        out = net(images)
        loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
        loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)
        loss = loss_l_pa12 + loss_c_pal2
        losses += loss.data[0]
        step += 1

    tloss = losses / step
    t2 = time.time()
    print('Timer: %.4f' % (t2 - t1))
    print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        print('Saving best state,epoch', epoch)
        torch.save(dsfd_net.state_dict(), os.path.join(
            save_to, 'dsfd.pth'))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': dsfd_net.state_dict(),
    }
    torch.save(states, os.path.join(save_to, 'dsfd_checkpoint.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()