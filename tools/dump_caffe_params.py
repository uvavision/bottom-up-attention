#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: dump-caffe-params.py
# $Date: Tue Mar 29 23:46:27 2016 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
import _init_paths
import sys
# caffe_root = '../caffe/'
# sys.path.insert(0, '../caffe/python/')
sys.path.insert(0, '../lib/')
sys.path.insert(0, './')
import caffe
import argparse

import sys
if sys.version_info.major < 2:
    import cPickle as pickle
else:
    import pickle

# Check object extraction
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
import cv2

GPU_ID = 1   # if we have multiple GPUs, pick one 
caffe.set_device(GPU_ID)  
caffe.set_mode_gpu()
net = None
cfg_from_file('../experiments/cfgs/faster_rcnn_end2end_resnet.yml')
weights = '../data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
prototxt = '../models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'

net = caffe.Net(prototxt, caffe.TEST, weights=weights)

rst = dict()
for name, p in net.params.items():
    W, b = p
    # if name.startswith('fc'):
    #     from IPython import embed; embed()
    rst['{}:W'.format(name)] = W.data
    rst['{}:b'.format(name)] = b.data
    print(name, W.data.shape, b.data.shape)

with open('example.npy', 'wb') as f:
    pickle.dump(rst, f, protocol=pickle.HIGHEST_PROTOCOL)
