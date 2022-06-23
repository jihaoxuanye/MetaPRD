import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from ..utils.data import Preprocessor
from ..utils.data import transforms as T
from mprd.gcn.pred_paiewise_related import PairsRelationship

def output_sim(inputs, pos_protomemory, neg_protomemory, targets):

    cp = pos_protomemory[targets]
    l_pos = (inputs.mm(cp.t())).diag()  # 64*1
    B = inputs.size(0)
    l_neg = inputs.mm(neg_protomemory.t())
    outputs = l_neg
    for b in range(B):
        outputs[b, targets[b]] = l_pos[b]

    return outputs

def g_b_backward(g_output_b, pos_protomemory, neg_protomemory, targets, b):
    proto_id = targets[b]
    neg_protomemory[proto_id] = pos_protomemory[proto_id]
    g_inputs_b = g_output_b.mm(neg_protomemory).view(-1)
    return g_inputs_b

def g_backward(g_outputs, pos_protomemory, neg_protomemory, targets):
    B = targets.size(0)
    g_inputs = list()
    for b in range(B):
        g_outputs_b = g_outputs[b].view(1, -1)
        g_inputs_b = g_b_backward(g_outputs_b, pos_protomemory, neg_protomemory, targets, b)
        g_inputs.append(g_inputs_b)

    g_inputs = torch.stack(g_inputs, dim=0)

    return g_inputs

class mixHard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, labels, features, pos_protomemory, neg_protomemory, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, labels, pos_protomemory, neg_protomemory)
        outputs = output_sim(inputs, pos_protomemory, neg_protomemory, labels)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, labels, pos_protomemory, neg_protomemory = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = g_backward(grad_outputs, pos_protomemory, neg_protomemory, labels)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, labels.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None, None, None

def mix_hard_contrastive(inputs, labels, protomemory, pos_protomemory, neg_protomemory, momentum=0.5):
    return mixHard.apply(inputs, labels, protomemory, pos_protomemory, neg_protomemory, torch.Tensor([momentum]).to(inputs.device))

class mixingContrastiveFeature(nn.Module):
    def __init__(self, index2targets=None, prior_index2tragets=None, extracted_features=None, alpha=0.2, use_gcn=False):

        super(mixingContrastiveFeature, self).__init__()
        self.index2targets = index2targets  # tensor  N
        self.prior_index2targets = prior_index2tragets
        self.extracted_features = extracted_features

        if use_gcn:
            self.pred_pairs_relate = PairsRelationship()
        else:
            self.pred_pairs_relate = None

        self.alpha = alpha
        self.use_gcn = use_gcn  # whether use gcn to pred the pair relationship

    def get_hard_samples(self, i_idx, i_tgt):

        labels_pair = self.index2targets[i_idx].expand_as(self.index2targets).eq(self.index2targets).float()
        prior_labels_pair = self.prior_index2targets[i_idx].expand_as(self.prior_index2targets).eq(self.prior_index2targets).float()

        diff_current_prior = labels_pair - prior_labels_pair
        positive_results = torch.nonzero(diff_current_prior > 0.5)
        negative_results = torch.nonzero(diff_current_prior < -0.5)

        return positive_results, negative_results

    def forward(self, inputs_q, protomemory, targets, indexes):

        inputs = inputs_q.clone().detach()
        pos_protomemory = protomemory.clone()
        neg_protomemory = protomemory.clone()
        nhard_protomemory = dict()  # dict

        batch_new_positive = list()
        batch_new_negative = list()

        for i, (idx, tgt) in enumerate(zip(indexes, targets)):
            # mixing positive prototype feature
            new_positive, new_negative = self.get_hard_samples(i_idx=idx, i_tgt=tgt)  # solved (use gcn for selecting refine positive)

            # pred pair-relationship of hard positives
            if new_positive.size(0) > 0 and self.use_gcn:
                pred_pos_relation = self.pred_pairs_relate.pair_relationship(img_idx=idx,
                                                                             new_idxes=new_positive,
                                                                             memory=self.extracted_features)

                new_positive = new_positive[(pred_pos_relation >= 0.5)]

            if new_positive.size(0) <= 0:
                new_positive = torch.nonzero(self.index2targets[idx.item() - 1].expand_as(self.index2targets).eq(self.index2targets).float() > 0.5)

            batch_new_positive.append(new_positive)

            # select a random positive feature (best)
            if True:
                ran_pos_idx = np.random.choice(new_positive.view(-1).cpu(), size=1)
                new_positive_feature = self.extracted_features[ran_pos_idx].squeeze(0)
                pos_protomemory[tgt] = self.alpha * new_positive_feature + (1. - self.alpha) * pos_protomemory[tgt]  # mixing random positive features

            # select a hardest positive feature
            if False:
                new_all_positive_features = self.extracted_features[new_positive.clone().cuda().view(-1)]
                new_hardest_positive_idx = new_positive[torch.argmin(inputs[i].view(1, -1).mm(new_all_positive_features.t())).item()]
                new_hardest_positive_feature = self.extracted_features[new_hardest_positive_idx].squeeze(0)
                pos_protomemory[tgt] = self.alpha * new_hardest_positive_feature + (1. - self.alpha) * pos_protomemory[tgt]  # mixing hardest positive features

            # calculate mean positive prototype features
            if False:
                new_all_positive_features = self.extracted_features[new_positive.clone().cuda().view(-1)]
                new_positive_prototype = torch.mean(new_all_positive_features, dim=0)
                new_positive_prototype /= new_positive_prototype.norm()
                pos_protomemory[tgt] = self.alpha * new_positive_prototype + (1. - self.alpha) * pos_protomemory[tgt]  # mixing mean positive prototype features

            # mixing negative prototype feature. it is hard to generate the specific negative memory, use jensen inequality
            if new_negative.size(0) > 0:
                if self.use_gcn:   # gcn negative
                    pred_neg_relation = self.pred_pairs_relate.pair_relationship(img_idx=idx,
                                                                             new_idxes=new_negative,
                                                                             memory=self.extracted_features)

                    new_negative = new_negative[(torch.tensor(pred_neg_relation).float() < 0.5)]  # pred pair-relationship of hard negatives

                nhard_protomemory[i] = self.extracted_features[new_negative].squeeze(1)

            batch_new_negative.append(new_negative)

            if self.use_gcn:
                self.pred_pairs_relate.collect_related_pairs(
                    memory=self.extracted_features, pred_hard_positive=new_positive,
                    pred_hard_negative=new_negative, img_tgt=tgt,
                    img_idx=idx, targets=targets, indexes=indexes
                )

        return pos_protomemory, neg_protomemory, nhard_protomemory, batch_new_positive, batch_new_negative


class Memory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, lamb=0.05, temp=0.05, momentum=0.2, use_hard=False, mix_module=None, nce_module=None):
        super(Memory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        if mix_module is not None:
            self.mix_module = mix_module
        if nce_module is not None:
            self.nce_module = nce_module

        self.lamb = lamb
        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets, indexes):

        # mixing hard positive features
        inputs = F.normalize(inputs, dim=1).cuda()
        pos_protomemory, neg_protomemory, nh_protomemory, new_positive_idx, new_negative_idx = self.mix_module(
            inputs_q=inputs, protomemory=self.features,
            targets=targets, indexes=indexes)

        outputs = mix_hard_contrastive(inputs=inputs, labels=targets, protomemory=self.features,
                                       pos_protomemory=pos_protomemory, neg_protomemory=neg_protomemory,
                                       momentum=self.momentum)

        # updating extracted features
        for indice, index in enumerate(indexes):
            self.mix_module.extracted_features[index] = inputs[indice].clone().detach()  # driectly update (current best)
            # momentum_update_vector = \
            #     self.alpha*self.mix_module.extracted_features[index] + (1.-self.alpha)*inputs[indice].clone().detach()  # momentum update
            # self.mix_module.extracted_features[index] = momentum_update_vector / momentum_update_vector.norm()

        outputs /= self.temp
        l_pos = torch.exp(outputs[:, targets].diag())
        l_neg = torch.exp(outputs).sum(dim=1)

        for ind, l_n in enumerate(l_neg):

            if ind in nh_protomemory:
                new_hard_negative_features = nh_protomemory[ind]
                sim_nh = inputs[ind].view(1, -1).mm(new_hard_negative_features.t()).view(-1)
                sim_nh /= self.temp
                logit_delta = torch.exp(sim_nh).sum()
                l_neg[ind] += self.lamb * logit_delta
                # l_neg[ind] = (1. - self.lamb) * l_neg[ind] + self.lamb * l_pos[ind] + self.lamb * logit_delta

        loss = -1*torch.log(l_pos / l_neg).mean()  # L_c

        return loss, nh_protomemory, new_positive_idx, new_negative_idx

# calculate the relative entropy of a positive pairs

normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

re_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

class RelativeEntropy(object):

    def __init__(self, index2targets, clusterset=None, tau=0.1, temp=0.05):
        self.index2targets = index2targets
        self.kl_div = nn.KLDivLoss(size_average=False, reduce=False)
        self.re_sampler = Preprocessor(dataset=clusterset, transform=re_transformer)
        self.tau = tau
        self.temp = temp

    def _mixing_relative_protomemory(self, f_out, f_hat, protomemory, img_indexes, sample_features, neg_hard_features, batch_pos_idx, batch_neg_idx, use_hard=True):
        '''
        :param f_out:
        :param f_hat:
        :param protomemory:
        :param img_indexes:
        :param sample_features:
        :param neg_hard_features: {in_batch_idx: all hard negative features of idx query}
        :param batch_pos_idx:
        :param batch_neg_idx: list() [[hard_neg_idxes_1], [hard_neg_idxes_2], ..., [hard_neg_idxes_batch]]
        :param use_hard:  if True, use hardest features to update the positive features; else, use random hard features to update
        :return:
        '''

        targets_h = self.index2targets[img_indexes]
        re_query_protomemory = protomemory.clone().detach()
        re_positive_protomemory = protomemory.clone().detach()

        for i, th in enumerate(targets_h):
            pos_idx = batch_pos_idx[i].view(-1)

            if use_hard:  # use hard features to update the feature memory
                f_pos = sample_features[pos_idx]
                sim_pos = (f_out[i].view(1, -1).mm(f_pos.t())).detach()
                hard_pos_idx = pos_idx[torch.argmin(sim_pos).item()]
            else:
                hard_pos_idx = np.random.choice(pos_idx.cpu(), size=1)

            hard_pos_features = sample_features[hard_pos_idx].detach().view(-1)
            re_query_protomemory[th] = self.tau * hard_pos_features + (1. - self.tau) * re_query_protomemory[th]
            re_query_protomemory[th] /= re_query_protomemory[th].norm()

            # mixing hard negative features of query sample to distribution for relative entropy
            if batch_neg_idx[i].size(0) > 0:
                th_negs = self.index2targets[batch_neg_idx[i]]
                query_neg_feats = neg_hard_features[i]
                for hn_idx, th_n in enumerate(th_negs):
                    hard_neg_feat = query_neg_feats[hn_idx].detach().view(-1)
                    if th_n < re_query_protomemory.size(0) and th_n != th:
                        re_query_protomemory[th_n] = self.tau * hard_neg_feat + (1. - self.tau) * re_query_protomemory[th_n]
                        re_query_protomemory[th_n] /= re_query_protomemory[th_n].norm()

            # mixing positive protomemory for relative entropy
            if use_hard:
                sim_pos = (f_hat[i].view(1, -1).mm(f_pos.t())).detach()
                easy_pos_idx = pos_idx[torch.argmin(sim_pos).item()]
            else:
                easy_pos_idx = np.random.choice(pos_idx.cpu(), size=1)

            easy_pos_features = sample_features[easy_pos_idx].detach().view(-1)
            re_positive_protomemory[th] = self.tau * easy_pos_features + (1. - self.tau) * re_positive_protomemory[th]
            re_positive_protomemory[th] /= re_positive_protomemory[th].norm()

            # mixing hard negative features of the positive sample to distribution for relative entropy
            if batch_neg_idx[i].size(0) > 0:
                th_negs = self.index2targets[batch_neg_idx[i]]
                positive_neg_feats = neg_hard_features[i]
                for hn_idx, th_n in enumerate(th_negs):
                    hard_neg_feat = positive_neg_feats[hn_idx].detach().view(-1)
                    if th_n < re_query_protomemory.size(0) and th_n != th:
                        re_positive_protomemory[th_n] = self.tau * hard_neg_feat + (1. - self.tau) * re_positive_protomemory[th_n]
                        re_positive_protomemory[th_n] /= re_positive_protomemory[th_n].norm()

        return re_query_protomemory, re_positive_protomemory

    def get_positive_samples(self, img_indexes, f_out=None, sample_features=None):

        targets_h = self.index2targets[img_indexes]
        pospair = list()
        for i, th in enumerate(targets_h):
            pos_idx = torch.nonzero(self.index2targets == th).view(-1)
            if pos_idx.size(0) > 1:
                # random positive
                ran_pos_idx = np.random.choice(pos_idx.cpu(), size=1)
                pospair.append([img_indexes[i].item(), ran_pos_idx.item()])
            else:
                pospair.append([img_indexes[i].item(), img_indexes[i].item()])

        pospair = np.array(pospair)
        positive_indices = pospair[:, 1].tolist()
        positive_samples = self.re_sampler.sampler(positive_indices)
        positive_samples = torch.stack(positive_samples).cuda()

        return positive_samples

    def relative_entropy_criterion(self, model, pos_samples, protomemory, \
                                   f_out, batch_pos_idx=None, batch_neg_idx=None, img_indexex=None, sample_features=None, neg_hard_features=None):
        '''
        :param model:
        :param pos_samples:
        :param protomemory:
        :param f_out:
        :param batch_pos_idx:
        :param batch_neg_idx:
        :param img_indexex:
        :param sample_features:
        :param neg_hard_features: dict() {in_batch_idx: all hard negative features of idx query}
        :return:
        '''

        f_hat = model(pos_samples)

        re_query_protomemory, re_positive_protomemory = \
            self._mixing_relative_protomemory(f_out=f_out, f_hat=f_hat,
                                              protomemory=protomemory, img_indexes=img_indexex,
                                              sample_features=sample_features, neg_hard_features=neg_hard_features,
                                              batch_pos_idx=batch_pos_idx, batch_neg_idx=batch_neg_idx)   # yes
        logits = F.softmax(f_out.mm(re_query_protomemory.t()) / self.temp, dim=1)
        logits_hat = f_hat.mm(re_positive_protomemory.t()) / self.temp

        logits_hat = F.log_softmax(logits_hat, dim=1)
        loss_kl = torch.mean(self.kl_div(logits_hat, logits).sum(dim=1))

        return loss_kl, f_hat

