#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020/09/09
author: relu
"""
import argparse
from data.widerface import WiderFace


root_dir = '/home/jovyan/jupyter/benchmark_images/faceu/face_detection/widerface'
def infer_args():
    parser = argparse.ArgumentParser(description=' inference')
    parser.add_argument('--data_dir',   type=str,   default=root_dir)
    parser.add_argument('--is_debug',   type=bool,  default=True)
    parser.add_argument('--batch_size', type=int,   default=32)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    dataset = WiderFace(args=infer_args())
    imginfo = dataset.__getitem__(4)
