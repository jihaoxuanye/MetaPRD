# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.backends import cudnn
from torch.autograd import Variable
from torch.nn import functional as F

from mprd import datasets
from mprd import models
from mprd.utils.data import transforms as T

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)

class MetaProcessor(Dataset):
    def __init__(self, metaset, root=None, transformer=None):
        super(MetaProcessor, self).__init__()
        self.metaset = metaset
        self.root = root
        self.transformer = transformer

    def __len__(self):
        return len(self.metaset)

    def __getitem__(self, idx):
        meta_fname, _, _ = self.metaset[idx]
        fpath = meta_fname

        if self.root is not None:
            fpath = osp.join(self.root, meta_fname)

        img = Image.open(fpath).convert('RGB')

        if self.transformer is not None:
            img1 = self.transformer(img)
            img2 = self.transformer(img)
            img3 = self.transformer(img)
            img4 = self.transformer(img)
            img5 = self.transformer(img)
            img6 = self.transformer(img)
            img7 = self.transformer(img)
            img8 = self.transformer(img)

        return img1, img2, img3, img4, img5, img6, img7, img8

def meta_data_loader(dataset, height, width, batch_size, worker, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    meta_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    meta_data = sorted(dataset)
    meta_dataset = MetaProcessor(metaset=meta_data, transformer=meta_transformer)
    meta_loader = DataLoader(dataset=meta_dataset, batch_size=5, num_workers=4, shuffle=False, pin_memory=True)

    return meta_loader

def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    print('Using pretrained model on ImageNet', model.pretrained)
    model.cuda()
    model = nn.DataParallel(model)
    return model

def main_worker(args):

    cudnn.benchmark = True
    print("==========\nArgs:{}\n==========".format(args))
    dataset = get_data(args.dataset, args.data_dir)

    random_meta_image = random.sample(dataset.train, 5)
    print('---------------------------------------------meta data-------------------------------------------------')
    for meta_img in random_meta_image:
        print(meta_img[0], meta_img[1])
    print('-------------------------------------------------------------------------------------------------------')

    meta_loader = meta_data_loader(dataset=random_meta_image, height=256, width=128, batch_size=5, worker=4)
    model = nn.DataParallel(create_model(args)).cuda()

    for inputs in meta_loader:
        meta_o = torch.stack([model(ipt) for ipt in inputs], dim=0).permute(1, 0, 2)
        # meat_feat_list = torch.stack([F.normalize((meta_o[0]+meta_o[1])/2, dim=-1),
        #                   F.normalize((meta_o[1]+meta_o[2])/2, dim=-1),
        #                   F.normalize((meta_o[2]+meta_o[3])/2, dim=-1),
        #                   F.normalize((meta_o[3]+meta_o[4])/2, dim=-1),
        #                   F.normalize((meta_o[4]+meta_o[0])/2, dim=-1)], dim=0)

    saved_path = './saved_model/pretrained/market_meta_data.pt'
    torch.save(meta_o, saved_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mixing hard contrastive features for unsupervised person re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.02,
                        help="learning rate")  # original learning rate is 0.00035
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='avg')
    parser.add_argument('--use-hard', action="store_true")
    main()