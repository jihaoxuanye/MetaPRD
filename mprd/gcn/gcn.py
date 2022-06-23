import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from mprd.gcn.pytorch_util import weights_init, gnn_spmm
from mprd.gcn.gnn_lib import GNNLIB
import math

class Classifier(nn.Module):
    def __init__(self, regression=False, whether_node_tag=True):
        super(Classifier, self).__init__()
        self.regression = regression
        model = DGCNN
        self.node_dist_featdim = 16
        self.node_tag_flag = whether_node_tag

        self.gnn = model(latent_dim=[512, 512, 1],
                         output_dim=0,
                         num_node_feats=16 + 2048,
                         num_edge_feats=0,
                         k=0,
                         conv1d_channels=[1024, 2048],
                         conv1d_kws=[0, 5],
                         conv1d_activation='ReLU')
        # out_dim = cmd_args.out_dim
        # if out_dim == 0:
        #     if cmd_args.gm == 'DGCNN':
        out_dim = 2048
            # else:
            #     out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(input_size=2048, hidden_size=1024, num_class=2,  with_dropout=True)
        self.k = 0

    def PrepareFeatureLabel(self, batch_graph):
        if self.regression:
            labels = torch.FloatTensor(len(batch_graph))
        else:
            labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
            subgraph_discrive_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        edge_feat_flag = False
        node_tag_flag = self.node_tag_flag

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags.tolist()
                subgraph_discrive_tag.append(batch_graph[i].node_tags.tolist())
            if node_feat_flag == True:
                tmp = batch_graph[i].node_features
                concat_feat.append(tmp)
            if edge_feat_flag == True:
                if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                    tmp = torch.from_numpy(batch_graph[i].edge_features).type('torch.FloatTensor')
                    concat_edge_feat.append(tmp)

        graph_discri_node_tag = []
        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            # node_tag = torch.zeros(n_nodes, 12)
            node_tag = torch.zeros(n_nodes, self.node_dist_featdim)
            node_tag.scatter_(1, concat_tag, 1)
            for ix, sdt in enumerate(subgraph_discrive_tag):
                # discri_node_tag = torch.zeros(len(sdt), 12)
                discri_node_tag = torch.zeros(len(sdt), self.node_dist_featdim)
                discri_node_tag.scatter_(1, torch.tensor(sdt).view(-1, 1), 1)
                graph_discri_node_tag.append(discri_node_tag)

        for iy in range(len(batch_graph)):
            batch_graph[iy].subgraph_node_tag = graph_discri_node_tag[iy]

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features

        if edge_feat_flag == True:
            edge_feat = torch.cat(concat_edge_feat, 0)

        # if cmd_args.mode == 'gpu':
        node_feat = node_feat.cuda()
        labels = labels.cuda()
        if edge_feat_flag == True:
            edge_feat = edge_feat.cuda()

        if edge_feat_flag == True:
            return node_feat, edge_feat, labels
        return node_feat, labels

    def forward(self, batch_graph, y=None):
        num_node_list = sorted([g.num_nodes for g in batch_graph])
        k_ = int(math.ceil(0.6*(len(num_node_list)))) - 1
        sort_pooling_k = max(num_node_list[k_], 10)  # to do
        self.gnn.k = sort_pooling_k
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return torch.stack([self.mlp(embed[i], y=y) for i in range(len(embed))])

    def output_features(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return embed, labels

class DGCNN(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 32, 32, 1], k=30, conv1d_channels=[16, 32], conv1d_kws=[0, 5], conv1d_activation='ReLU'):
        print('Initializing DGCNN')
        super(DGCNN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats + num_edge_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        self.advpool1d = nn.AdaptiveMaxPool1d(1)

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))

        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)

        graph_info = [GNNLIB.PrepareSparseMatrices([gl]) for gl in graph_list]
        H = []
        for ind, (n2n_sp, e2n_sp, subg_sp) in enumerate(graph_info):
            if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
                n2n_sp = n2n_sp.cuda()
                e2n_sp = e2n_sp.cuda()
                subg_sp = subg_sp.cuda()
                node_degs = node_degs.cuda()
            node_feat = Variable(torch.cat((graph_list[ind].subgraph_node_tag.cuda(), graph_list[ind].node_features), dim=1))  # n1*2060
            if edge_feat is not None:
                edge_feat = Variable(edge_feat)
                if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
                    edge_feat = edge_feat.cuda()
            n2n_sp = Variable(n2n_sp)
            e2n_sp = Variable(e2n_sp)
            subg_sp = Variable(subg_sp)
            # node_degs = Variable(node_degs)
            node_degs = Variable(torch.tensor(graph_list[ind].degs).view(-1, 1) + 1.)
            node_degs = node_degs.cuda()

            h = self.sortpooling_embedding(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs)
            H.append(h)
        # H = torch.stack(H)
        return H

    def sortpooling_embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs):
        ''' if exists edge feature, concatenate to node feature vector '''
        if edge_feat is not None:
            #input_edge_linear = self.w_e2l(edge_feat)
            input_edge_linear = edge_feat
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        ''' graph convolution layers '''
        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer  # Y = (A + I) * X
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W
            normalized_linear = node_linear.div(node_degs.float())  # Y = D^-1 * Y
            cur_message_layer = torch.tanh(normalized_linear)
            # cur_message_layer = torch.relu(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)

        to_sort = cur_message_layer[:, -1]
        k = self.k if self.k <= node_feat.size(0) else node_feat.size(0)
        _, top_indices = to_sort.topk(k)
        sort_pooling_graph = cur_message_layer.index_select(0, top_indices)
        if k < self.k:
            to_pad = torch.zeros(self.k - k, self.total_latent_dim)
            to_pad = to_pad.cuda()
            sort_pooling_graph = torch.cat((sort_pooling_graph, to_pad), dim=0)  ###

        sort_pooling_graph = sort_pooling_graph.unsqueeze(0)
        to_conv1d = sort_pooling_graph.view(1, 1, -1)
        conv1d_res = self.maxpool1d(
            self.conv1d_activation(
                self.conv1d_params1(to_conv1d)
            )
        )
        conv1d_res = self.advpool1d(
            self.conv1d_activation(
            self.conv1d_params2(conv1d_res)
        ))
        to_dense = conv1d_res.view(1, -1)
        return self.conv1d_activation(to_dense)

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, acc
        else:
            return logits
