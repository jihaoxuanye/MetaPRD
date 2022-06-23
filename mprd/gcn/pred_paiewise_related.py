import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
import tqdm
from copy import deepcopy

from mprd.gcn.gcn import Classifier
from mprd.gcn.util_gcn import \
    sample_neg, link2graph, subgraph_extraction_labeling, GNNGraph, test_links, train_link, meta_link2graph
from torch.utils.data import Dataset, DataLoader

class PairRelateGenerator(Dataset):

    def __init__(self, pairdata):
        self.dataset = pairdata

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indice):

        node_left, node_right, pair_label = self.dataset[indice]
        node_pair = torch.tensor([node_left, node_right])

        return node_pair, pair_label

def get_pair_loader(pair_data):

    pair_dataset = PairRelateGenerator(pair_data)
    pair_loader = DataLoader(dataset=pair_dataset, batch_size=128,
                             num_workers=4, shuffle=True,
                             pin_memory=True, drop_last=True)

    return pair_loader


class PairsRelationship(object):

    def __init__(self):
        super(PairsRelationship, self).__init__()

        self.pred = Classifier(whether_node_tag=True).cuda()

        loaded_state_dict = torch.load(
            '/home/yons/PycharmProject/cluster-contrast-reid-main/clustercontrast/gcn/saved_model/pretrained/pretrained_meta_market_gcn.pth.tar')

        state_dict = dict()
        for key in loaded_state_dict: state_dict[key[7:]] = loaded_state_dict[key]
        self.pred.load_state_dict(state_dict)

        params = [{"params": [value]} for _, value in self.pred.named_parameters() if value.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.02, weight_decay=5e-4)
        self.collected_pair = list()

    def pair_relationship(self, img_idx, new_idxes, memory, topk=8):
        img_indexes = [img_idx.item()] + new_idxes.view(-1).tolist()

        sample_pair = torch.tensor(sample_neg(img_indexes))
        g_lists = []

        for pos_idx, signal_sample_pair in enumerate(sample_pair):
            subgraphs = link2graph(signal_sample_pair, memory, max_train_num=topk)
            gs, subgraph_labels, nodes_idx = subgraph_extraction_labeling([signal_sample_pair], subgraphs)
            g_lists += [GNNGraph(g, 1, n_label, memory[n_idx])
                      for g, n_label, n_idx in zip(gs, subgraph_labels, nodes_idx)]

        links_pred, link_logits = test_links(g_list=g_lists, classifier=self.pred)

        return links_pred

    def collect_related_pairs(self, memory, pred_hard_positive, pred_hard_negative, img_tgt, img_idx, targets, indexes):

        # collecting positive sample pairs
        if pred_hard_positive.size(0) >= 1:
            pos_idxes = torch.cat([img_idx.unsqueeze(0), pred_hard_positive.view(-1).cpu()], dim=0)
            selected_pos_pair = np.append(np.random.choice(pos_idxes, size=2, replace=False), 1).tolist()
            self.collected_pair.append(selected_pos_pair)
        else:
            pos_idxes = indexes[torch.nonzero((targets == img_tgt) > 0.)].cpu()
            selected_pos_pair = np.append(np.random.choice(pos_idxes, size=2), 1).tolist()
            self.collected_pair.append(selected_pos_pair)

        # collecting negative sample pairs
        if pred_hard_negative.size(0) >= 1:
            pos_idxes = torch.cat([img_idx.unsqueeze(0), pred_hard_positive.view(-1).cpu()], dim=0)
            selected_pos_idx = np.random.choice(pos_idxes, size=1)
            selected_pos_feature = memory[selected_pos_idx]
            negative_features = memory[pred_hard_negative.view(-1)]
            selected_neg_idx = pred_hard_negative[torch.argmax(selected_pos_feature.mm(negative_features.t()))].item()
            selected_neg_pair = np.append(np.append(selected_pos_idx, selected_neg_idx), 0).tolist()
            self.collected_pair.append(selected_neg_pair)
        else:
            pos_idxes = torch.cat([img_idx.unsqueeze(0), pred_hard_positive.view(-1).cpu()], dim=0)
            selected_pos_idx = np.random.choice(pos_idxes, size=1)
            selected_pos_feature = memory[selected_pos_idx]
            neg_idxes = indexes[torch.nonzero((targets != img_tgt) > 0.)].view(-1).cpu()
            negative_features = memory[neg_idxes]
            selected_neg_idx = neg_idxes[torch.argmax(selected_pos_feature.mm(negative_features.t()))].item()
            selected_neg_pair = np.append(np.append(selected_pos_idx, selected_neg_idx), 0).tolist()
            self.collected_pair.append(selected_neg_pair)

    def train_loop(self, memory, topk=8):
        pair_loader = get_pair_loader(self.collected_pair)
        pbar = tqdm.tqdm(pair_loader)

        for idx, (node_pairs, node_labels) in enumerate(pbar):

            training_data = list()

            for pair in node_pairs:
                subgraphs = link2graph(pair, memory, topk)
                gs, subgraph_labels, nodes_idx = subgraph_extraction_labeling(pair.unsqueeze(0).tolist(), subgraphs)
                g_list = [GNNGraph(g, 1, n_label, memory[n_idx])
                          for g, n_label, n_idx in zip(gs, subgraph_labels, nodes_idx)]
                training_data += g_list

            logt = train_link(g_list=training_data, classifier=self.pred, label=None)
            logt = logt.squeeze(1)
            y = Variable(node_labels.cuda())
            loss_g = F.nll_loss(logt, y)

            pred = logt.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(pred.size(0))
            true_pos = torch.nonzero(y.data.view_as(pred).cpu() == 1)[:, 0]
            recall = pred[true_pos].sum().item() / float(true_pos.size(0))

            self.optimizer.zero_grad()
            loss_g.backward()
            self.optimizer.step()

            pbar.set_description('gcn_loss: '+format(loss_g.item(), '.4f')+' gcn_acc: '+format(acc, '.4f')+' gcn_recall: '+format(recall, '.4f'))

            # print('gcn_loss: '+format(loss_g.item(), '.4f')+'  gcn_acc: '+format(acc, '.4f')+'  gcn_recall: '+format(recall, '.4f'))

    def train_reptile_loop(self, memory, topk=8, innerepoch=3, inner_stepsize=0.02):   # GCN reptile oprimization strategy
        pair_loader = get_pair_loader(self.collected_pair)
        pbar = tqdm.tqdm(pair_loader)

        for idx, (node_pairs, node_labels) in enumerate(pbar):

            training_data = list()

            gcn_weights_before = deepcopy(self.pred.state_dict())

            for _ in range(innerepoch):
                for pair in node_pairs:
                    subgraphs = link2graph(pair, memory, topk)
                    gs, subgraph_labels, nodes_idx = subgraph_extraction_labeling(pair.unsqueeze(0).tolist(), subgraphs)
                    g_list = [GNNGraph(g, 1, n_label, memory[n_idx])
                              for g, n_label, n_idx in zip(gs, subgraph_labels, nodes_idx)]
                    training_data += g_list

                logt = train_link(g_list=training_data, classifier=self.pred, label=None)
                logt = logt.squeeze(1)
                y = Variable(node_labels.cuda())

                del training_data
                training_data = list()

                loss_g = F.nll_loss(logt, y)
                loss_g.backward()

                for param in self.pred.parameters():
                    param.data -= inner_stepsize * param.grad.data

            pred = logt.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(pred.size(0))
            true_pos = torch.nonzero(y.data.view_as(pred).cpu() == 1)[:, 0]
            recall = pred[true_pos].sum().item() / float(true_pos.size(0))

            gcn_weights_after = self.pred.state_dict()

            self.optimizer.zero_grad()

            for param_idx, (name, param) in enumerate(zip(gcn_weights_before, self.pred.parameters())):
                delta_param = gcn_weights_after[name] - gcn_weights_before[name]
                param.grad.data = -delta_param
            # loss_g.backward()

            self.optimizer.step()

            pbar.set_description('gcn_loss: '+format(loss_g.item(), '.4f')+' gcn_acc: '+format(acc, '.4f')+' gcn_recall: '+format(recall, '.4f'))

            # print('gcn_loss: '+format(loss_g.item(), '.4f')+'  gcn_acc: '+format(acc, '.4f')+'  gcn_recall: '+format(recall, '.4f'))