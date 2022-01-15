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
        if sum(gt==1) == 0:
            _loss = - (1.0 - gt) * torch.log(1.0 - cas + 0.00001)
            return _loss
        r = float(len(gt)) / sum(gt==1)
        if r == 1:
            coef_0 = torch.tensor(0).cuda()
            coef_1 = torch.tensor(1).cuda()
        else:
            coef_0 = 0.5 * r / (r - 1)
            coef_1 = coef_0 * (r - 1)
        _loss_1 = - coef_1 * gt * torch.log(cas + 0.00001)
        _loss_0 = - coef_0 * (1.0 - gt) * torch.log(1.0 - cas + 0.00001)
        _loss = _loss_1 + _loss_0
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

    def balanced_ce(self, gt, cas, gt_class, loss_type='bce'):
        '''
        loss_type = 'bce', 'mse', 'ce'
        '''
        act_loss = torch.tensor(0.).cuda()
        bkg_loss = torch.tensor(0.).cuda()
        act_count = 0
        bkg_count = 0
        for i in range(cas.shape[0]):
            for j in range(cas.shape[-1]):
                gt_ = gt[i, :, j]
                cas_ = cas[i, :, j]
                if self.thres_down < 0:
                    gt_ = (gt_ > self.thres).float().cuda()
                else:
                    _gt = torch.ones_like(gt_).cuda() * -1
                    _gt[gt_ > self.thres] = 1
                    _gt[gt_ <= self.thres_down] = 0
                    cas_ = cas_[_gt >= 0]
                    gt_ = _gt[_gt >= 0]
                if gt_class[i, j] > 0:
                    if loss_type == 'bce':
                        _loss = self.BCE(gt_, cas_)
                    elif loss_type == 'mse':
                        _loss = torch.norm(cas_ - gt_, p=2)
                    else:
                        _loss = self.bi_loss(gt_, cas_)
                    act_loss = act_loss + torch.mean(_loss)
                    act_count += 1
                else:
                    if self.bkg_lmbd > 0:
                        _gt = torch.zeros_like(cas_).cuda()
                        if loss_type == 'bce':
                            _loss = self.BCE(_gt, cas_)
                        elif loss_type == 'mse':
                            _loss = torch.norm(cas_ - _gt, p=2)
                        else:
                            _loss = self.bi_loss(_gt, cas_)
                        bkg_loss = bkg_loss + torch.mean(_loss)
                    bkg_count += 1
        act_loss = act_loss / act_count
        bkg_loss = bkg_loss / bkg_count

        return act_loss, bkg_loss

    def forward(self, score_act, score_bkg, feat_act, feat_bkg, label,
                gt, cas, score_act_t=None, cas_t=None):
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
            loss_sup_act, loss_sup_bkg = self.balanced_ce(gt, cas, label, 'bce')
            loss_sup_act = self.lmbd * loss_sup_act
            loss_sup_bkg = self.lmbd * self.bkg_lmbd * loss_sup_bkg
            loss_sup = loss_sup_act + loss_sup_bkg
            loss_total = loss_total + loss_sup
            loss["loss_sup_act"] = loss_sup_act
            loss["loss_sup_bkg"] = loss_sup_bkg
            loss["loss_sup"] = loss_sup_bkg
            print("loss_sup_act", (loss_sup_act).detach().cpu().item())
            print("loss_sup_bkg", (loss_sup_bkg).detach().cpu().item())
            print("loss_sup", (loss_sup).detach().cpu().item())
        
        if cas_t is not None:
            loss_sup_act, loss_sup_bkg = self.balanced_ce(gt, cas, label, 'mse')
            loss_ema_f = self.gamma_f * (loss_sup_act + self.bkg_lmbd * loss_sup_bkg)
            loss_ema_c = self.gamma_c * torch.norm(score_act - score_act_t, p=2).mean()
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
        score_act_t, _, _, _, _, _, sup_cas_softmax_t = net_teacher(_data)
        cost, loss = criterion(score_act, score_bkg, feat_act, feat_bkg, _label,
                               _gt, sup_cas_softmax, score_act_t, sup_cas_softmax_t)
    else:
        cost, loss = criterion(score_act, score_bkg, feat_act, feat_bkg, _label,
                               _gt, sup_cas_softmax)

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
