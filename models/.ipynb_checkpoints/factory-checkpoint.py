#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from models.DSFD_vgg import build_net_vgg
from models.DSFD_resnet import build_net_resnet


def build_net(phase = 'train', num_classes = 2, model = 'vgg'):
    if phase not in ['train', 'test']:
        print("ERROR: Phase: " + phase + " not recognized")
        return

    if model not in ['vgg', 'resnet']:
        print("ERROR: model:" + model + " not recognized")
        return

    if model == 'vgg':
        return build_net_vgg(phase, num_classes)
    else:
        return build_net_resnet(phase, num_classes, model)



def basenet_factory(model='vgg'):
	if model=='vgg':
		basenet = 'vgg16_reducedfc.pth'

	elif 'resnet' in model:
		basenet = '{}.pth'.format(model)
	return basenet

