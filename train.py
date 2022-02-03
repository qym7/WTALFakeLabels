import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

import utils


class GCNN_loss(nn.Module):
    def __init__(self, gcnn_weight):
        super(GCNN_loss, self).__init__()
        self.gcnn_weight = gcnn_weight

    def contrastive_loss(self, node, pos_nodes, neg_nodes):
        pos_sim = torch.norm(pos_nodes - node, p=2, dim=1)
        neg_sim = torch.norm(neg_nodes - node, p=2, dim=1)
        pos_loss = torch.tensor(0).cuda()
        neg_loss = torch.tensor(0).cuda()
        # 在这里更改similarity的格式
        similarity = nn.CosineSimilarity(dim=0)
        if pos_nodes.shape[0] != 0:
            # print('pos', torch.argmax(pos_sim), torch.max(nn.functional.gumbel_softmax(pos_sim)))
            pos_sample = pos_nodes[torch.argmax(pos_sim).detach()]
            # pos_sample = torch.matmul(nn.functional.gumbel_softmax(pos_sim), pos_nodes)  # choose most different positive sample
            pos_loss = 1 - similarity(node, pos_sample)
        if neg_nodes.shape[0] != 0:
            # print('neg', torch.argmax(neg_sim),  torch.max(nn.functional.gumbel_softmax(neg_sim)))
            # neg_sample = torch.matmul(nn.functional.gumbel_softmax(-neg_sim), neg_nodes)  # choose most similar negetive sample
            neg_sample = neg_nodes[torch.argmin(neg_sim).detach()]
            neg_loss = similarity(node, neg_sample)

        return pos_loss + neg_loss

    def forward(self, nodes, nodes_label):
        loss = torch.tensor(0).float().cuda()
        total_count = torch.tensor(0).cuda()
        for i in range(len(nodes)):  # iterate different class: 因为每个class的node数量不同，不可并行操作
            node = nodes[i]
            node_label = nodes_label[i]  # N * 2048
            pos_node = node[node_label==2]
            neg_node = node[node_label==0]
            for j, n in enumerate(node):
                if node_label[j] == 0:
                    loss += self.contrastive_loss(n, neg_node, pos_node)
                else:
                    loss += self.contrastive_loss(n, pos_node, neg_node)
                total_count += 1
                
        return loss / total_count * self.gcnn_weight


class UM_loss(nn.Module):
    def __init__(self, alpha, beta, lmbd, neg_lmbd, bkg_lmbd, margin, thres, thres_down,
                 gamma_f, gamma_c, N):
        super(UM_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lmbd = lmbd
        self.neg_lmbd = neg_lmbd
        self.bkg_lmbd = bkg_lmbd
        self.margin = margin
        self.thres = thres
        self.thres_down = thres_down
        self.gamma_f = gamma_f
        self.gamma_c = gamma_c
        self.N = N
        self.ce_criterion = nn.BCELoss()

    def BCE(self, gt, cas):
        cas = torch.sigmoid(cas)

        coef_0 = torch.ones(gt.shape[0]).cuda()
        coef_1 = torch.zeros(gt.shape[0]).cuda()
        r = torch.zeros(gt.shape[0]).cuda()
        act_pos = torch.any(gt==1, dim=1)
        r[act_pos] = gt[act_pos].shape[-1] / (gt[act_pos]==1).sum(dim=-1)

        coef_0[r==1] = 0
        coef_1[r==1] = 1
        coef_0[r>1] = 0.5 * r[r>1] / (r[r>1] - 1)
        coef_1[r>1] = coef_0[r>1] * (r[r>1] - 1)

        _loss_1 = - coef_1 * (gt * torch.log(cas + 0.00001)).mean()
        _loss_0 = - coef_0 * ((1.0 - gt) * torch.log(1.0 - cas + 0.00001)).mean()
        _loss = (_loss_1 + _loss_0).mean()

        return _loss

    def bi_loss(self, gt: torch.Tensor, logits):
        """
        gt: (batch, time_frame, 20)
        cas: (batch, time_frame, 20)
        separate the loss for gt == 1 and gt == 0, calculate the mean for each.
        """
        logits_pos = logits[gt.bool()]
        logits_neg = logits[~gt.bool()]
        if len(logits_pos) > 0:
            loss_pos = F.binary_cross_entropy(torch.sigmoid(logits_pos).float(),
                                              gt[gt.bool()].float(), reduction='mean')
        else:
            loss_pos = torch.tensor(0.).cuda()
        if len(logits_neg) > 0:
            loss_neg = F.binary_cross_entropy(torch.sigmoid(logits_neg).float(),
                                              gt[~gt.bool()].float(), reduction='mean')
        else:
            loss_neg = torch.tensor(0.).cuda()
        return loss_pos + self.neg_lmbd * loss_neg

    def balanced_ce(self, gt, cas, label, loss_type='bce'):
        '''
        loss_type = 'bce', 'mse', 'ce'
        gt: BS * 750 * 20
        label: BS * 20
        '''
        if self.thres_down < 0:
            gt = (gt > self.thres).float().cuda()
            gt = torch.permute(gt, (0, 2, 1))
            gt[~label.bool()] = 0
            gt = torch.permute(gt, (0, 2, 1))
        else:
            gt = torch.ones_like(gt).cuda() * -1
            gt[gt > self.thres] = 1
            gt[gt <= self.thres_down] = 0
            cas = cas[gt >= 0]
            gt = gt[gt >= 0]

        act_loss = torch.tensor(0.).cuda()
        bkg_loss = torch.tensor(0.).cuda()

        gt_channel_pos = torch.any(gt == 1, dim=1)
        cas_gt_channel = torch.permute(cas, (0, 2, 1))[gt_channel_pos]
        gt_gt_channel = torch.permute(gt, (0, 2, 1))[gt_channel_pos]
        cas_non_gt_channel = torch.permute(cas, (0, 2, 1))[~gt_channel_pos]
        gt_non_gt_channel = torch.permute(gt, (0, 2, 1))[~gt_channel_pos]

        if loss_type == 'bce':
            act_loss = self.BCE(gt_gt_channel, cas_gt_channel)
            bkg_loss = self.BCE(gt_non_gt_channel, cas_non_gt_channel)
        elif loss_type == 'be':
            act_loss = self.bi_loss(gt_gt_channel, cas_gt_channel)
            bkg_loss = self.bi_loss(gt_non_gt_channel, cas_non_gt_channel)
        else:
            act_loss = torch.norm(gt_gt_channel - cas_gt_channel, p=2).mean()
            bkg_loss = torch.norm(gt_non_gt_channel - cas_non_gt_channel, p=2).mean()

        # act_loss = act_loss / gt_channel_pos.sum()
        # bkg_loss = bkg_loss / (~gt_channel_pos).sum()

        return act_loss, bkg_loss

    def forward(self, score_act, score_bkg, feat_act, feat_bkg, label,
                gt, cas, nodes=None, nodes_label=None,
                score_act_t=None, score_bkg_t=None, cas_t=None, step=0):
        loss = {}
 
        label = label / torch.sum(label, dim=1, keepdim=True)

        loss_cls = self.ce_criterion(score_act, label)

        label_bkg = torch.ones_like(label).cuda()
        label_bkg /= torch.sum(label_bkg, dim=1, keepdim=True)
        loss_be = self.ce_criterion(score_bkg, label_bkg)

        loss_act = self.margin - torch.norm(torch.mean(feat_act, dim=1), p=2, dim=1)
        loss_act[loss_act < 0] = 0
        loss_bkg = torch.norm(torch.mean(feat_bkg, dim=1), p=2, dim=1)

        loss_um = torch.mean((loss_act + loss_bkg) ** 2)

        loss_total = loss_cls + self.alpha * loss_um + self.beta * loss_be

        # if cas is not None:
        #     loss_sup_act, loss_sup_bkg = self.balanced_ce(gt, cas, 'bce')
        #     loss_sup_act = self.lmbd * loss_sup_act
        #     loss_sup_bkg = self.lmbd * self.bkg_lmbd * loss_sup_bkg
        #     loss_sup = loss_sup_act + loss_sup_bkg
        #     # loss_total = loss_total + loss_sup * torch.pow(input=torch.tensor(0.98), exponent=step/50)
        #     loss_total = loss_total + loss_sup
        #     loss["loss_sup_act"] = loss_sup_act
        #     loss["loss_sup_bkg"] = loss_sup_bkg
        #     print("loss_sup_act", (loss_sup_act).detach().cpu().item())
        #     print("loss_sup_bkg", (loss_sup_bkg).detach().cpu().item())
        
        # if cas_t is not None:
        #     loss_ema_f = self.gamma_f * torch.norm(cas - cas_t, p=2).mean()
        #     loss_ema_c = self.gamma_c * torch.norm(score_act - score_act_t, p=2).mean() +\
        #         self.gamma_c * torch.norm(score_bkg - score_bkg_t, p=2).mean()
        #     loss_total = loss_total + loss_ema_f + loss_ema_c
        #     loss["loss_ema_f"] = loss_ema_f
        #     loss["loss_ema_c"] = loss_ema_c
        #     print("loss_ema_f", (loss_ema_f).detach().cpu().item())
        #     print("loss_ema_c", (loss_ema_c).detach().cpu().item())

        loss["loss_cls"] = loss_cls
        loss["loss_be"] = loss_be
        loss["loss_um"] = loss_um
        loss["loss_total"] = loss_total
        print("loss_cls", loss_cls.detach().cpu().item())
        print("loss_be", (self.beta * loss_be).detach().cpu().item())
        print("loss_um", (self.alpha * loss_um).detach().cpu().item())
        print("loss_total", loss_total.detach().cpu().item())

        return loss_total, loss


def train(net, gcnn, loader_iter, optimizer, optimizer_gcnn, criterion, criterion_gcnn, logger, step, net_teacher, m, nodes_dict):
    net.train()

    _data, _label, _gt, vid_names, _, index = next(loader_iter)
    N = _data.shape[1]
    _data = _data.cuda()  # reshaped in net
    _gt = _gt.cuda()  # reshaped in net
    _label = _label.reshape(-1, _label.shape[-1]).cuda()

    _data, nodes, nodes_label, vids_label = gcnn(_data, _gt, index, vid_names)
    cost_gcnn = criterion_gcnn(nodes, nodes_label)

    ######################## ema code to be deleted
    for i in range(len(nodes)):
        class_node = nodes[i]
        vid_label = vids_label[i]
        for j in range(N):
            vid_cls_name = vid_names[i][j]+f'_{index[i]}'
            if vid_cls_name not in nodes_dict:
                nodes_dict[vid_cls_name] = nodes[i][vid_label==j].detach()
            else:
                nodes[i][vid_label==j] = nodes[i][vid_label==j] * 0.01 + nodes_dict[vid_cls_name] * 0.99
                nodes_dict[vid_cls_name] = nodes[i][vid_label==j].detach()
    ########################

    _data = _data.detach()
    _gt = _gt.reshape(-1, _gt.shape[-2], _gt.shape[-1]).detach()
    score_act, score_bkg, feat_act, feat_bkg, _, _, sup_cas_softmax = net(_data)

    cost, loss = criterion(score_act, score_bkg, feat_act, feat_bkg, _label,
                            _gt, sup_cas_softmax, step=step)
    loss['loss_gcnn'] = cost_gcnn
    print("loss_gcnn", cost_gcnn.detach().cpu().item())

    optimizer.zero_grad()
    optimizer_gcnn.zero_grad()
    cost_gcnn.backward()
    cost.backward()
    optimizer.step()
    optimizer_gcnn.step()

    if net_teacher is not None:
        # update teacher parameters by EMA
        student_params = OrderedDict(net.named_parameters())
        teacher_params = OrderedDict(net_teacher.named_parameters())

        # check if both model contains the same set of keys
        assert student_params.keys() == teacher_params.keys()

        for name, param in student_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            teacher_params[name] = teacher_params[name] * m + (1 - m) * param

        student_buffers = OrderedDict(net.named_buffers())
        teacher_buffers = OrderedDict(net_teacher.named_buffers())

        # check if both model contains the same set of keys
        assert student_buffers.keys() == teacher_buffers.keys()

        for name, buffer in student_buffers.items():
            teacher_buffers[name] = teacher_buffers[name] * m + (1 - m) * buffer

    for key in loss.keys():
        logger.log_value(key, loss[key].cpu().item(), step)
        

