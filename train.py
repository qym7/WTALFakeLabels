import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

import utils


class UM_loss(nn.Module):
    def __init__(self, alpha, beta, lmbd, neg_lmbd, bkg_lmbd, margin, thres, thres_down,
                 gamma_f, gamma_c):
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
        self.ce_criterion = nn.BCELoss()

    def BCE(self, gt, cas):
        cas = torch.sigmoid(cas)

        coef_0 = torch.ones(gt.shape[0]).cuda()
        coef_1 = torch.zeros(gt.shape[0]).cuda()
        r = torch.zeros(gt.shape[0]).cuda()
        pos_gt = torch.any(gt==1, dim=1)
        r[pos_gt] = gt.shape[-1] / (gt[pos_gt]==1).sum(dim=-1)

        coef_0[r==1] = 0
        coef_1[r==1] = 1
        coef_0[r>1] = 0.5 * r[r>1] / (r[r>1] - 1)
        coef_1[1>1] = coef_0[r>1] * (r[r>1] - 1)
            
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

    def balanced_ce(self, gt, cas, loss_type='bce'):
        '''
        loss_type = 'bce', 'mse', 'ce'
        '''
        if self.thres_down < 0:
            gt = (gt > self.thres).float().cuda()
        else:
            gt = torch.ones_like(gt).cuda() * -1
            gt[gt > self.thres] = 1
            gt[gt <= self.thres_down] = 0
            cas = cas[gt >= 0]
            gt = gt[gt >= 0]
        
        act_loss = torch.tensor(0.).cuda()
        bkg_loss = torch.tensor(0.).cuda()
        
        gt_channel_pos = torch.any(gt == 1, dim=2)
        cas_gt_channel = cas[gt_channel_pos]
        gt_gt_channel = gt[gt_channel_pos]
        cas_non_gt_channel = cas[~gt_channel_pos]
        gt_non_gt_channel = gt[~gt_channel_pos]

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
                gt, cas, score_act_t=None, score_bkg_t=None, cas_t=None, step=0):
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

        if cas is not None:
            loss_sup_act, loss_sup_bkg = self.balanced_ce(gt, cas, 'bce')
            loss_sup_act = self.lmbd * loss_sup_act
            loss_sup_bkg = self.lmbd * self.bkg_lmbd * loss_sup_bkg
            loss_sup = loss_sup_act + loss_sup_bkg
            # loss_total = loss_total + loss_sup * torch.pow(input=torch.tensor(0.98), exponent=step/50)
            loss_total = loss_total + loss_sup
            loss["loss_sup_act"] = loss_sup_act
            loss["loss_sup_bkg"] = loss_sup_bkg
            loss["loss_sup"] = loss_sup_bkg
            print("loss_sup_act", (loss_sup_act).detach().cpu().item())
            print("loss_sup_bkg", (loss_sup_bkg).detach().cpu().item())
            print("loss_sup", (loss_sup).detach().cpu().item())
        
        if cas_t is not None:
            loss_ema_f = self.gamma_f * torch.norm(cas - cas_t, p=2).mean()
            loss_ema_c = self.gamma_c * torch.norm(score_act - score_act_t, p=2).mean() +\
                self.gamma_c * torch.norm(score_bkg - score_bkg_t, p=2).mean()
            loss_total = loss_total + loss_ema_f + loss_ema_c
            loss["loss_ema_f"] = loss_ema_f
            loss["loss_ema_c"] = loss_ema_c
            print("loss_ema_f", (loss_ema_f).detach().cpu().item())
            print("loss_ema_c", (loss_ema_c).detach().cpu().item())

        loss["loss_cls"] = loss_cls
        loss["loss_be"] = loss_be
        loss["loss_um"] = loss_um
        loss["loss_total"] = loss_total
        print("loss_cls", loss_cls.detach().cpu().item())
        print("loss_be", (self.beta * loss_be).detach().cpu().item())
        print("loss_um", (self.alpha * loss_um).detach().cpu().item())
        print("loss_total", loss_total.detach().cpu().item())

        return loss_total, loss

def train(net, loader_iter, optimizer, criterion, logger, step, net_teacher, m):
    net.train()

    _data, _label, _gt, _, _ = next(loader_iter)

    _data = _data.cuda()
    _label = _label.cuda()
    if _gt is not None:
        _gt = _gt.cuda()

    optimizer.zero_grad()

    score_act, score_bkg, feat_act, feat_bkg, _, _, sup_cas_softmax = net(_data)

    if net_teacher is not None:
        score_act_t, score_bkg_t, _, _, _, _, sup_cas_softmax_t = net_teacher(_data)
        cost, loss = criterion(score_act, score_bkg, feat_act, feat_bkg, _label,
                               _gt, sup_cas_softmax, score_act_t, score_bkg_t, sup_cas_softmax_t, step=step)
    else:
        cost, loss = criterion(score_act, score_bkg, feat_act, feat_bkg, _label,
                               _gt, sup_cas_softmax, step=step)

    cost.backward()
    optimizer.step()

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
        

