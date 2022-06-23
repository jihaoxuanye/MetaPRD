from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter


class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None, re_crit=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.re_crit = re_crit

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes, img_indexes = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)
            loss_mh, neg_hard_protomemory, batch_pos_idx, batch_neg_idx = self.memory(f_out, labels, img_indexes)

            # calculate relative entropy criterion
            pos_samples = self.re_crit.get_positive_samples(img_indexes=img_indexes, f_out=f_out, sample_features=self.memory.mix_module.extracted_features)
            loss_kl, f_hat = self.re_crit.relative_entropy_criterion(model=self.encoder,
                                                              pos_samples=pos_samples,
                                                              protomemory=self.memory.features, f_out=f_out,
                                                              batch_pos_idx=batch_pos_idx, batch_neg_idx=batch_neg_idx,
                                                              img_indexex=img_indexes,
                                                              sample_features=self.memory.mix_module.extracted_features, neg_hard_features=neg_hard_protomemory)

            loss = loss_mh + loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

        # training the gcn which following the training epoch of the CNN
        if self.memory.mix_module.use_gcn:
            # self.memory.mix_module.pred_pairs_relate.train_loop(memory=self.memory.mix_module.extracted_features)
            self.memory.mix_module.pred_pairs_relate.train_reptile_loop(memory=self.memory.mix_module.extracted_features)
            self.memory.mix_module.pred_pairs_relate.collected_pair = list()


    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes, img_indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), img_indexes

    def _forward(self, inputs):
        return self.encoder(inputs)

