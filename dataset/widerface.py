#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020/09/09
author: relu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as data
from dataset.augmentations import preprocess

from IPython import embed


class WiderFace(data.Dataset):

    def __init__(self, args, mode = 'train'):
        super(WiderFace, self).__init__()
        self.args = args
        self.mode = mode
        self._loading_anno()
        
    
    def _loading_anno(self):
        self.rows = pd.read_csv(os.path.join(self.args.data_dir, 'wider_face_anno.csv'))
        self.rows = self.rows.sample(frac=1.0, replace=False)
        if self.args.is_debug:
            self.rows = self.rows.iloc[3 * self.args.batch_size, :]
            print('Attention! debug mode is going ...')
        self.rows = self.rows.to_numpy().tolist()
        print('there are %04d rows were loaded' % len(self.rows))


    def __len__(self):
        return len(self.rows)

    
    def __getitem__(self, index):
        
        while True:
            img_path = os.path.join(self.args.data_dir, 'images', self.rows[index][0])
            img = Image.open(img_path)
            if img.mode == 'L':
                img = img.convert('RGB')
            img_width, img_height = img.size
            anno = self.annotransform(np.array(eval(self.rows[index][1])), img_width, img_height)
            img, target = preprocess(img, anno, self.mode) 
            if len(target) > 0:
                break
            else:
                index = random.randrange(0, len(self.rows))
        return img, target
        
    def annotransform(self, anno, img_w, img_h):
        anno[:, 0] /= img_w
        anno[:, 1] /= img_h
        anno[:, 2] /= img_w
        anno[:, 3] /= img_h
        anno[:, 4] = anno[:, 4].astype(np.int)
        return anno


def detection_collate(batch):
    imgs, targets = [], []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)
    return (torch.stack(imgs, 0), targets)
