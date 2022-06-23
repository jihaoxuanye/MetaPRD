# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.backends import cudnn
from torch.autograd import Variable
from torch.nn import functional as F

from mprd import datasets
from mprd import models
from mprd.evaluators import Evaluator, extract_features
from mprd.utils.data import transforms as T
from mprd.utils.data.preprocessor import Preprocessor, Preprocessor_train
from mprd.gcn.gcn import Classifier
from mprd.gcn.util_gcn import train_link2graph, meta_link2graph, subgraph_extraction_labeling, GNNGraph, train_link

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_pretrain_loader(dataset, height, width, batch_size, worker, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    pair_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir, transform=pair_transformer),
        batch_size=batch_size, num_workers=worker,
        shuffle=True, pin_memory=True, drop_last=True
    )

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    print('Using pretrained model on ImageNet', model.pretrained)
    model.cuda()
    model = nn.DataParallel(model)
    return model

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)

def main_worker(args):
    cudnn.benchmark = True

    # generating feature memory
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)

    # Create model
    model = nn.DataParallel(create_model(args)).cuda()  # feature embedding function (fixing it at the gcn pretrain period)
    gcn = nn.DataParallel(Classifier(whether_node_tag=True)).cuda()

    # GCN Optimizer
    params = [{"params": [value]} for _, value in gcn.named_parameters() if value.requires_grad]
    gcn_optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)

    memory_loader = get_test_loader(dataset, args.height, args.width,
                                     args.batch_size, args.workers, testset=sorted(dataset.train))

    features, _ = extract_features(model, memory_loader, print_freq=50)
    features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0).cuda()

    pair_loader = get_pretrain_loader(dataset, args.height, args.width,
                                      args.batch_size, args.workers, trainset=sorted(dataset.train))

    pbar = tqdm.tqdm(pair_loader)

    meta_data = torch.load('saved_model/pretrained/market_meta_data.pt').contiguous().view(-1, 2048)

    # meta data to train the model
    innerepoch = 3
    meta_batch = 10
    inner_stepsize = args.lr

    meta_training_data = []
    meta_training_labels = []
    gcn_weights_before = deepcopy(gcn.state_dict())

    for _ in range(innerepoch):
        for _ in range(meta_batch):
            # select a positive meta data
            meta_id = np.random.choice(np.arange(0, 5), size=1)
            meta_obj = np.random.choice(np.arange(0, 8), size=1)
            selected_meta = 8 * meta_id + meta_obj
            meta_pos_list = np.arange(8 * meta_id, 8 * meta_id + 8)
            meta_pos_subgraph, meta_pos_pair = meta_link2graph(meta_left=selected_meta.tolist(),
                                                       meta_pos_list=meta_pos_list, memory=features,
                                                       meta_memory=meta_data)
            meta_pos_gs, meta_pos_subgraph_labels, meta_pos_nodes_idx = subgraph_extraction_labeling(meta_pos_pair, meta_pos_subgraph)
            cated_features = torch.cat((features, meta_data), dim=0)
            meta_pos_g_list = [GNNGraph(g, 1, n_label, cated_features[n_idx])
                           for g, n_label, n_idx in zip(meta_pos_gs, meta_pos_subgraph_labels, meta_pos_nodes_idx)]
            meta_training_data += meta_pos_g_list
            meta_training_labels.append(1)

            # select a negative meta data
            meta_neg_subgraph, meta_neg_pair = meta_link2graph(meta_left=selected_meta.tolist(),
                                                       meta_pos_list=meta_pos_list, memory=features,
                                                       meta_memory=meta_data, positive_flag=False)
            meta_neg_gs, meta_neg_subgraph_labels, meta_neg_nodes_idx = subgraph_extraction_labeling(meta_neg_pair, meta_neg_subgraph)
            meta_neg_g_list = [GNNGraph(g, 1, n_label, cated_features[n_idx])
                           for g, n_label, n_idx in zip(meta_neg_gs, meta_neg_subgraph_labels, meta_neg_nodes_idx)]
            meta_training_data += meta_neg_g_list
            meta_training_labels.append(0)

        # training one batch
        meta_logt = train_link(g_list=meta_training_data, classifier=gcn, label=None)
        meta_logt = meta_logt.squeeze(1)
        meta_training_labels = torch.tensor(meta_training_labels)
        meta_y = Variable(meta_training_labels).cuda()

        del meta_training_data, meta_training_labels
        meta_training_data, meta_training_labels = list(), list()

        loss_meta = F.nll_loss(meta_logt, meta_y)
        loss_meta.backward()

        for param_idx, param in enumerate(gcn.parameters()):
            param.data -= inner_stepsize * param.grad.data

    gcn_weights_after = gcn.state_dict()
    gcn.load_state_dict({name: gcn_weights_before[name] + inner_stepsize*(gcn_weights_after[name] - gcn_weights_before[name])
                           for name in gcn_weights_before})

    # training the gcn on the given data
    for ind, inputs in enumerate(pbar):
        imgs, _, _, _, img_indexes = inputs
        o = model(imgs)
        f_out = o.clone().detach()

        pre_training_data = []
        pre_training_labels = []

        # create graph structure (create a positive and a negative pair) overall meta
        gcn_weights_before = deepcopy(gcn.state_dict())

        for _ in range(innerepoch):
            for idx, img_idx in enumerate(img_indexes):
                # select a positive pairs
                train_pair = torch.tensor([img_idx, img_idx]).unsqueeze(0)
                subgraphs = train_link2graph(train_pos=train_pair, memory=features, max_train_num=8, f_out=f_out[idx])

                gs, subgraph_labels, nodes_idx = subgraph_extraction_labeling(train_pair, subgraphs)
                g_list = [GNNGraph(g, 1, n_label, features[n_idx])
                          for g, n_label, n_idx in zip(gs, subgraph_labels, nodes_idx)]

                pre_training_data += g_list
                pre_training_labels.append(1)

                # select a negative pairs
                bias_sim = torch.zeros(size=(f_out.size(0),)).cuda()
                bias_sim[idx] = 10.
                hard_batch_neg_idx = img_indexes[torch.argmax(f_out[idx].view(1, -1).mm(f_out.t()) - bias_sim)]
                train_pair = torch.tensor([img_idx, hard_batch_neg_idx]).unsqueeze(0)
                subgraphs = train_link2graph(train_pos=train_pair, memory=features, max_train_num=8, f_out=f_out[idx])

                gs, subgraph_labels, nodes_idx = subgraph_extraction_labeling(train_pair, subgraphs)
                g_list = [GNNGraph(g, 1, n_label, features[n_idx])
                          for g, n_label, n_idx in zip(gs, subgraph_labels, nodes_idx)]

                pre_training_data += g_list
                pre_training_labels.append(0)

            logt = train_link(g_list=pre_training_data, classifier=gcn, label=None)
            logt = logt.squeeze(1)
            pre_training_labels = torch.tensor(pre_training_labels)
            y = Variable(pre_training_labels).cuda()

            del pre_training_data, pre_training_labels
            pre_training_data, pre_training_labels = list(), list()

            loss_g = F.nll_loss(logt, y)
            loss_g.backward()
            for param in gcn.parameters():
                param.data -= inner_stepsize * param.grad.data

        gcn_weights_after = gcn.state_dict()

        pred = logt.data.max(1, keepdim=True)[1]
        acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(pred.size(0))
        true_pos = torch.nonzero(y.data.view_as(pred).cpu() == 1)[:, 0]
        recall = pred[true_pos].sum().item() / float(true_pos.size(0))

        gcn_optimizer.zero_grad()

        for param_idx, (name, param) in enumerate(zip(gcn_weights_before, gcn.parameters())):
            delta_param = gcn_weights_after[name] - gcn_weights_before[name]
            param.grad.data = -delta_param

        # loss_g.backward()

        gcn_optimizer.step()

        # print('idx:', ind, 'loss_g:', loss_g.item(), 'acc:', acc, 'recall:', recall)
        pbar.set_description(
            'gcn_loss: ' + format(loss_g.item(), '.4f') + ' gcn_acc: ' + format(acc, '.4f') + ' gcn_recall: ' + format(
                recall, '.4f'))

    # saving pretrained gcn model
    fpath = './saved_model/pretrained/pretrained_meta_market_gcn.pth.tar'
    torch.save(gcn.state_dict(), fpath)
    print('pretrain process finish...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MPRD for unsupervised person re-ID")
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
                        help="GCN learning rate")
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