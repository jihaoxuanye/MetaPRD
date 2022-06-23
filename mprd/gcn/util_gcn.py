import numpy as np
import torch
import scipy
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import networkx as nx
import copy
# from .data_util import *
from torch.autograd import Variable
from torch.nn import functional as F

def sample_neg(pos_idx):
    perm = pos_idx[1:]
    train_pos = [[pos_idx[0], p] for p in perm]
    return train_pos

def link2graph(train_pos, memory, max_train_num, node_id=None, second_order=False, K=8):
    if not second_order:
        train_pos = train_pos.unsqueeze(0)

        left_node = train_pos[:, 0]
        left_vec = memory[left_node]
        left_sim = left_vec.mm(memory.t())
        _, left_idx_sorted = torch.sort(left_sim, dim=1, descending=True)
        left_idx_sorted = left_idx_sorted[:, :max_train_num]
        left_idx_sorted = torch.cat((torch.stack([left_node]).cuda().t(),
                                     left_idx_sorted[:, 1:]), dim=1).view(-1)
        left_link = sample_neg(left_idx_sorted)

        right_node = train_pos[:, 1]
        right_vec = memory[right_node]
        right_sim = right_vec.mm(memory.t())
        _, right_idx_sorted = torch.sort(right_sim, dim=1, descending=True)
        right_idx_sorted = right_idx_sorted[:, :max_train_num]

        subgarphs = list()
        for rid, r in enumerate(right_idx_sorted):
            # single_right_link = []
            single_right_link = [[r_right, r[0]] for r_right in r[1:]]
            subgarphs.append(torch.unique(torch.tensor(left_link[:K] + single_right_link[:K]), sorted=False, dim=0))
        # subgarphs = torch.tensor(subgarphs)
        graph_link_labels = []
        if node_id is not None:
            for idx, singel_pair in enumerate(train_pos):
                left_node, right_node = singel_pair[0], singel_pair[1]
                if node_id[left_node.item()] == node_id[right_node.item()]:
                    link_label = 1
                else:
                    link_label = 0
                graph_link_labels.append(link_label)

            return subgarphs, graph_link_labels

        return subgarphs
    else:
        left_link = train_pos
        # left_link = left_link[: max_train_num]
        train_pos = torch.tensor(train_pos)
        right_link = list()
        right_node = train_pos[:, 1]
        right_vec = memory[right_node]
        right_sim = right_vec.mm(memory.t())
        _, right_idx_sorted = torch.sort(right_sim, dim=1, descending=True)
        right_idx_sorted = right_idx_sorted[:, :max_train_num]
        subgarphs = list()
        second_order_left_node = right_idx_sorted[:, :4]
        second_order_left_link = list()
        for solis in second_order_left_node:
            second_order_left_link += [[solis_left, solis[0]] for solis_left in solis[1:]]
        # subgarphs = list()
        for rid, r in enumerate(right_idx_sorted):
            single_right_link = [[r_right, r[0]] for r_right in r[1:]]
            second_order_right_node = r[1:]
            _, second_order_right_idx_sorted = torch.sort(
                memory[second_order_right_node].mm(memory.t()),
                descending=True,
                dim=1
            )
            second_order_right_idx_sorted = second_order_right_idx_sorted[:, :4]
            second_single_right_link = list()
            for soris in second_order_right_idx_sorted:
                second_single_right_link += [[soris[0], soris_right] for soris_right in soris[1:]]
            subgarphs.append(torch.unique(torch.tensor(second_order_left_link + left_link + single_right_link + second_single_right_link), sorted=False, dim=0))

        return subgarphs

def link2graph2(train_pos, cur_feat, img_id, max_train_num, node_id=None, memory_copy=None, second_order=False, K=8):
    if not second_order:
        memory_copy[img_id] = cur_feat
        left_link = train_pos
        train_pos = torch.tensor(train_pos)
        right_link = list()
        right_node = train_pos[:, 1]
        right_vec = memory_copy[right_node]
        right_sim = right_vec.mm(memory_copy.t())
        _, right_idx_sorted = torch.sort(right_sim, dim=1, descending=True)
        right_idx_sorted = right_idx_sorted[:, :max_train_num]
        subgarphs = list()
        for rid, r in enumerate(right_idx_sorted):
            # single_right_link = []
            single_right_link = [[r_right, r[0]] for r_right in r[1:]]
            subgarphs.append(torch.unique(torch.tensor(left_link[:K] + single_right_link[:K]), sorted=False, dim=0))
        # subgarphs = torch.tensor(subgarphs)
        graph_link_labels = []
        if node_id is not None:
            for idx, singel_pair in enumerate(train_pos):
                left_node, right_node = singel_pair[0], singel_pair[1]
                if node_id[left_node.item()] == node_id[right_node.item()]:
                    link_label = 1
                else:
                    link_label = 0
                graph_link_labels.append(link_label)

            return subgarphs, graph_link_labels

        return subgarphs, memory_copy
    else:
        memory_copy[img_id] = cur_feat
        left_link = train_pos
        train_pos = torch.tensor(train_pos)
        right_link = list()
        right_node = train_pos[:, 1]
        right_vec = memory_copy[right_node]
        right_sim = right_vec.mm(memory_copy.t())
        _, right_idx_sorted = torch.sort(right_sim, dim=1, descending=True)
        right_idx_sorted = right_idx_sorted[:, :max_train_num]
        subgraphs = list()
        second_order_left_node = right_idx_sorted[:, :4]
        second_order_left_link = list()
        for solis in second_order_left_node:
            second_order_left_link += [[solis_left, solis[0]] for solis_left in solis[1:]]
        for rid, r in enumerate(right_idx_sorted):
            single_right_link = [[r_right, r[0]] for r_right in r[1:]]
            second_order_right_node = r[1:]
            _, second_order_right_idx_sorted = torch.sort(
                memory_copy[second_order_right_node].mm(memory_copy.t()),
                descending=True,
                dim=1
            )
            second_order_right_idx_sorted = second_order_right_idx_sorted[:, :4]
            second_single_right_link = list()
            for soris in second_order_right_idx_sorted:
                second_single_right_link += [[soris[0], soris_right] for soris_right in soris[1:]]
            subgraphs.append(torch.unique(torch.tensor(second_order_left_link + left_link + single_right_link + second_single_right_link), sorted=False, dim=0))

        return subgraphs, memory_copy

def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels > 1e6] = 0  # set inf labels to 0
    labels[labels < -1e6] = 0  # set -inf labels to 0
    return labels

def subgraph_extraction_labeling(train_pos, subgraphs):
    subgraph_labels = []
    gs = []
    nodes_idx = []
    for link, subgraph in zip(train_pos, subgraphs):
        subgraph = subgraph.tolist()
        # link = link.tolist()
        link = torch.tensor(link).tolist()
        if link in subgraph:
            subgraph.remove(link)
        subgraph = [link] + subgraph
        subgraph_copy = torch.unique(torch.tensor(subgraph))
        new_graph_node = dict()
        max_idx = 0
        subgraph = np.array(subgraph)
        for graph_idx, subgraph_node in enumerate(subgraph_copy):
            new_graph_node[subgraph_node.item()] = graph_idx
            if max_idx <= graph_idx:
                max_idx = graph_idx
        new_graph = np.array([[new_graph_node[subgraph[i, 0]], new_graph_node[subgraph[i, 1]]] for i in range(len(subgraph))])
        subgraph = ssp.csc_matrix((np.ones(len(subgraph)), (new_graph[:, 0], new_graph[:, 1])),
                                  shape=(max_idx+1, max_idx+1))
        labels = node_label(subgraph)
        labels = np.clip(labels, a_min=0, a_max=15)
        g = nx.from_scipy_sparse_matrix(subgraph)
        subgraph_labels.append(labels)
        node_idx = subgraph_copy.tolist()
        if g.has_edge(0, 1):
            g.remove_edge(0, 1)
        gs.append(g)
        nodes_idx.append(node_idx)
    return gs, subgraph_labels, nodes_idx

class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)  # self.node_features.requires_grad=True
        self.degs = list(dict(g.degree).values())
        self.subgraph_node_tag = None

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])

        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert (type(edge_features.values()[0]) == np.ndarray)
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)

def test_links(g_list, classifier):
    test_graph = g_list
    classifier.eval()
    logits = classifier(test_graph).squeeze(1)
    pos_link_idx = torch.exp(logits)[:, 1] > 0.5
    return pos_link_idx, torch.exp(logits)

def train_link(g_list, classifier, label):
    classifier.train()
    train_graph = g_list
    logits = classifier(train_graph, y=label)
    return logits

def check_pos(pair, symmetry, memory_copy, k):
    if pair[1].item() in symmetry.tolist(): return True
    sim_r2l = (memory_copy[pair[1]].view(1, -1)).mm(memory_copy.t())
    _, r2l_sorted_index = torch.sort(sim_r2l, dim=1, descending=True)
    r2l_sorted_index = r2l_sorted_index[0, :k]
    if pair[0].item() in r2l_sorted_index: return True
    return False

def createMultiLabel(epoch, train_pos, links_pred, train_pos2, link_pred1, symmetry, memory_copy, k):
    reviewer = dict()
    pos_link = list()
    train_idx = list()

    for idx, (pair, linkp) in enumerate(zip(train_pos2, link_pred1)):
        reviewer[pair[1].item()] = linkp.item()
    # gcn_link = []

    for idx, (pair, linkp) in enumerate(zip(train_pos, links_pred)):
        if pair[1].item() in reviewer and linkp.item() == reviewer[pair[1].item()]:
            if linkp == 1:
                # a = 1
                pos_link.append([idx, pair[0].item(), pair[1].item()])
                # gcn_link.append([idx, pair[0].item(), pair[1].item()])
            elif linkp.item() == 0 and idx == 0:
                if check_pos(pair, symmetry, memory_copy, k):
                    pos_link.append([idx, pair[0].item(), pair[1].item()])
                    train_idx.append([idx, pair[0].item(), pair[1].item(), 0.8])
                else:
                    train_idx.append([idx, pair[0].item(), pair[1].item(), 0.2])
        elif pair[1].item() in reviewer and linkp.item() != reviewer[pair[1].item()]:
            if check_pos(pair, symmetry, memory_copy, k):
                pos_link.append([idx, pair[0].item(), pair[1].item()])
                train_idx.append([idx, pair[0].item(), pair[1].item(), 0.8])
            else:
                train_idx.append([idx, pair[0].item(), pair[1].item(), 0.2])

    return pos_link, train_idx

def train_link2graph(train_pos, memory, max_train_num, f_out=None, second_order=False):
    if not second_order:
        train_pos = torch.tensor(train_pos)

        right_node = train_pos[:, 1]
        right_vec = memory[right_node]
        right_sim = right_vec.mm(memory.t())
        _, right_idx_sorted = torch.sort(right_sim, dim=1, descending=True)
        right_idx_sorted = right_idx_sorted[:, :max_train_num]

        left_node = train_pos[:, 0]
        if f_out is None:
            left_vec = memory[left_node]
        else:
            left_vec = f_out

        left_sim = left_vec.view(1, -1).mm(memory.t())
        _, left_idx_sorted = torch.sort(left_sim, dim=1, descending=True)
        left_idx_sorted = left_idx_sorted[:, :max_train_num]
        left_idx_sorted = torch.cat((torch.stack([left_node]).cuda(), left_idx_sorted[:, 1:]), dim=1).view(-1)
        left_link = sample_neg(left_idx_sorted)

        subgarphs = list()
        for rid, r in enumerate(right_idx_sorted):
            # single_right_link = []
            single_right_link = [[r_right, r[0]] for r_right in r[1:]]
            subgarphs.append(torch.unique(torch.tensor(left_link + single_right_link), sorted=False, dim=0))
        return subgarphs
    else:
        left_link = train_pos
        train_pos = torch.tensor(train_pos)
        right_link = list()
        right_node = train_pos[:, 1]
        right_vec = memory[right_node]
        right_sim = right_vec.mm(memory.t())
        _, right_idx_sorted = torch.sort(right_sim, dim=1, descending=True)
        right_idx_sorted = right_idx_sorted[:, :max_train_num]
        left_node = train_pos[:, 0]
        left_vec = memory[left_node]
        left_sim = left_vec.mm(memory.t())
        _, left_idx_sorted = torch.sort(left_sim, dim=1, descending=True)
        left_idx_sorted = left_idx_sorted[:, :max_train_num]
        left_idx_sorted = torch.cat((torch.stack([left_node]).cuda(), left_idx_sorted[:, 1:]), dim=1).view(-1)
        left_link = sample_neg(left_idx_sorted, max_train_num)
        second_order_left_node = left_idx_sorted[:4]
        second_order_left_link = list()
        second_order_left_link += [[solis_left, second_order_left_node[0]] for solis_left in second_order_left_node[1:]]
        subgarphs = list()
        for rid, r in enumerate(right_idx_sorted):
            single_right_link = [[r_right, r[0]] for r_right in r[1:]]
            second_order_right_node = r[1:]
            _, second_order_right_idx_sorted = torch.sort(
                memory[second_order_right_node].mm(memory.t()),
                descending=True,
                dim=1
            )
            second_order_right_idx_sorted = second_order_right_idx_sorted[:, :4]
            second_order_right_link = list()
            for soris in second_order_right_idx_sorted:
                second_order_right_link += [[soris[0], soris_right] for soris_right in soris[1:]]

            subgarphs.append(torch.unique(torch.tensor(second_order_left_link + left_link + single_right_link + second_order_right_link), sorted=False,
                                          dim=0))
        return subgarphs

def meta_link2graph(meta_left, meta_pos_list, memory, meta_memory, max_train_nums=8, positive_flag=True):
    num_features = memory.size(0)
    left_vec = meta_memory[meta_left]
    left_sim = left_vec.view(1, -1).mm(meta_memory.t())
    _, left_idx_sorted = torch.sort(left_sim, dim=1, descending=True)
    left_idx_sorted = left_idx_sorted[:, :max_train_nums].cpu()
    left_idx_sorted = torch.cat((torch.tensor([meta_left]), left_idx_sorted[:, 1:]), dim=1).view(-1)
    left_idx_sorted = left_idx_sorted + num_features
    left_link = sample_neg(left_idx_sorted)

    if positive_flag:
        meta_right = np.random.choice(meta_pos_list, size=1)
        node_pair = torch.tensor([meta_left, meta_right]).view(1, -1) + num_features
        right_vec = meta_memory[meta_right]
        right_sim = right_vec.view(1, -1).mm(meta_memory.t())
        _, right_idx_sorted = torch.sort(right_sim, dim=1, descending=True)
        right_idx_sorted = right_idx_sorted[:, :max_train_nums]
        right_idx_sorted = right_idx_sorted + num_features

        subgraphs = list()
        for rid, r in enumerate(right_idx_sorted):
            signal_right_link = [[r_right, r[0]] for r_right in r[1:]]
            subgraphs.append(torch.unique(torch.tensor(left_link + signal_right_link),
                                          sorted=False, dim=0))

        return subgraphs, node_pair

    if not positive_flag:
        meta_neg_list = list(set(np.arange(meta_memory.size(0)).tolist()) - set(meta_pos_list.tolist()))
        meta_neg_memory = meta_memory[meta_neg_list]
        meta_right = [meta_neg_list[torch.argmax(left_vec.view(1, -1).mm(meta_neg_memory.t())).item()]]
        node_pair = torch.tensor([meta_left, meta_right]).view(1, -1) + num_features

        right_vec = meta_memory[meta_right]
        right_sim = right_vec.view(1, -1).mm(meta_memory.t())
        _, right_idx_sorted = torch.sort(right_sim, dim=1, descending=True)
        right_idx_sorted = right_idx_sorted[:, :max_train_nums]
        right_idx_sorted = right_idx_sorted + num_features

        subgraphs = list()
        for rid, r in enumerate(right_idx_sorted):
            signal_right_link = [[r_right, r[0]] for r_right in r[1:]]
            subgraphs.append(torch.unique(torch.tensor(left_link + signal_right_link), sorted=False, dim=0))

        return subgraphs, node_pair

def pretrain_link2graph(train_pos, memory, out_i=None, max_train_num=None, node_id=None):
    left_link = train_pos
    train_pos = torch.tensor(train_pos)
    right_link = list()
    right_node = train_pos[:, 1]
    if right_node < memory.size(0):
        right_vec = memory[right_node]
    else:
        right_vec = out_i.unsqueeze(0)
    right_sim = right_vec.mm(memory.t())
    _, right_idx_sorted = torch.sort(right_sim, dim=1, descending=True)
    right_idx_sorted = right_idx_sorted[:, :max_train_num]
    right_idx_sorted = torch.cat((torch.stack([right_node]).cuda(), right_idx_sorted[:, 1:]), dim=1)

    left_node = train_pos[:, 0]
    left_vec = memory[left_node]
    left_sim = left_vec.mm(memory.t())
    _, left_idx_sorted = torch.sort(left_sim, dim=1, descending=True)
    left_idx_sorted = left_idx_sorted[:, :max_train_num]
    left_idx_sorted = torch.cat((torch.stack([left_node]).cuda(), left_idx_sorted[:, 1:]), dim=1).view(-1)
    left_link = sample_neg(left_idx_sorted)

    subgarphs = list()
    for rid, r in enumerate(right_idx_sorted):
        # single_right_link = []
        single_right_link = [[r_right, r[0]] for r_right in r[1:]]
        subgarphs.append(torch.unique(torch.tensor(left_link + single_right_link), sorted=False, dim=0))
    # subgarphs = torch.tensor(subgarphs)
    graph_link_labels = []
    if node_id is not None:
        for idx, singel_pair in enumerate(train_pos):
            left_node, right_node = singel_pair[0], singel_pair[1]
            if node_id[left_node.item()] == node_id[right_node.item()]:
                link_label = 1
            else:
                link_label = 0
            graph_link_labels.append(link_label)

        return subgarphs, graph_link_labels

    return subgarphs

def gcn_loss(target, model, memory, feat, feat_j, gcn, img, t=0.6):

    out_cnn = feat_j
    batch_sim = memory[target].mm(memory[target].t())
    batch_sim_withoutdiag = batch_sim*(1. - torch.eye(img.size(0)).cuda()) + 1e-8
    neg_img_idx = torch.max(batch_sim_withoutdiag, dim=1)[1]
    batch_train_g_list = list()
    train_label = []

    for i in range(target.size(0)):
        train_pos = [target[i], target[i] + memory.size(0)]
        train_neg = [target[i], target[neg_img_idx[i]]]
        train = [train_pos, train_neg]
        train_label.append(1)
        train_label.append(0)
        train_subgraphs = [pretrain_link2graph([train[ix]], memory, out_cnn[i], 8)[0] for ix in range(len(train))]
        train_gs, train_subgraph_label, train_nodes_idx = \
            subgraph_extraction_labeling(train, train_subgraphs)
        train_g_list = list()
        for idx, (g, n_label, n_idx) in enumerate(zip(train_gs, train_subgraph_label, train_nodes_idx)):
            node_features = list()
            cur_idx = target[i].item()
            for ix in range(len(n_idx)):
                if n_idx[ix] < memory.size(0) and n_idx[ix] != cur_idx:
                    node_features += [memory[n_idx[ix]]]
                elif n_idx[ix] == cur_idx:
                    node_features += [feat[i]]
                else:
                    node_features += [out_cnn[i]]
            node_features = torch.stack(node_features)
            train_g_list.append(GNNGraph(g, train_label[idx], n_label, node_features))
        # train_g_list = [GNNGraph(g, train_label[idx], n_label, torch.stack([memory[n_idx[ix]] if n_idx[ix]<memory.size(0) else out_cnn[i] for ix in range(len(n_idx))]))\
        #                 for idx, (g, n_label, n_idx) in enumerate(zip(train_gs, train_subgraph_label, train_nodes_idx))]
        batch_train_g_list += [[train_g_list[ind], train_g_list[ind].label] for ind in range(len(train_g_list))]

    batch_traingcn = batch_train_g_list
    training_data = [tgcn[0] for tgcn in batch_traingcn]
    training_label = [tgcn[1] for tgcn in batch_traingcn]
    training_label = torch.tensor(training_label).cuda()

    log = train_link(g_list=training_data, classifier=gcn, label=None)
    log = log.squeeze(1)
    y = Variable(training_label)
    pred = log.data.max(1, keepdim=True)[1]
    acc = pred.eq(y.data.view_as(pred)).cpu().sum().item()/float(pred.size(0))
    loss_gcn = F.nll_loss(log, y)
    return loss_gcn, acc

def pretrain_loss(logits, pids, output_i, output_j, epoch, model, memory, gcn, criterion, img):
    loss1 = criterion(logits, pids, [], output_i, output_j, epoch=epoch, data_memory=memory)
    loss2, acc = gcn_loss(
        target=pids,
        model=model,
        memory=memory,
        gcn=gcn,
        feat=output_i,
        feat_j = output_j,
        img=img
    )
    loss = loss1 + loss2
    return loss, acc
