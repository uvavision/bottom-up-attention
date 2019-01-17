#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: dump-caffe-params.py
# $Date: Tue Mar 29 23:46:27 2016 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
import sys
caffe_root = '../caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import argparse

import sys
if sys.version_info.major < 2:
    import cPickle as pickle
else:
    import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='prototxt')
    parser.add_argument(dest='model')
    parser.add_argument(dest='output')
    args = parser.parse_args()

    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    rst = dict()
    for name, p in net.params.items():
        W, b = p
        # if name.startswith('fc'):
        #     from IPython import embed; embed()
        rst['{}:W'.format(name)] = W.data
        rst['{}:b'.format(name)] = b.data
        print(name, W.data.shape, b.data.shape)

    with open(args.output, 'wb') as f:
        pickle.dump(rst, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()

# vim: foldmethod=marker
