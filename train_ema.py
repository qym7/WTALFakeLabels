'''
Author: your name
Date: 2021-12-25 17:33:51
LastEditTime: 2021-12-26 12:02:48
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /yimingqin/code/WTAL-Uncertainty-Modeling/train.py
'''
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

import utils


class UM_loss(nn.Module):
    def __init__(self, alpha, beta, lmbd, gamma, margin, thres):
        super(UM_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lmbd = lmbd
        self.gamma = gamma
        self.margin = margin
        self.thres = thres
        self.ce_criterion = nn.BCELoss()
        self.l2_criterion = nn.MSELoss()
    
    def balanced_ce(self, gt, cas):
        loss = 0
        count = 0
        pmask = (gt > self.thres).float().cuda()
        for i in range(cas.shape[0]):
            for j in range(cas.shape[-1]):
                if pmask[i, :, j].sum() > 0:
                    r = sum(pmask[i, :, j]==1) / float(pmask.shape[1])
                    coef_0 = 0.5 * r / (r - 1)
                    coef_1 = coef_0 * (r - 1)
                    # _loss = coef_1 * pmask[i, :, j] * torch.log(cas[i, :, j] + 0.00001) +\
                    #         coef_0 * (1.0 - pmask[i, :, j]) * torch.log(1.0 - cas[i, :, j] + 0.00001)
                    _loss = torch.norm(cas[i, :, j] - pmask[i, :, j], p=2)
                    loss = loss + torch.mean(_loss)
                    count += 1
        loss = loss / count

        return loss

    def forward(self, score_act, score_bkg, feat_act, feat_bkg, label,
                gt, sup_cas, cas_s, cas_t):
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

        if sup_cas is not None:
            loss_sup = self.balanced_ce(gt, sup_cas)
            loss["loss_sup"] = loss_sup
            print("loss_sup", (self.lmbd*loss_sup).detach().cpu().item())
            loss_total = loss_total + self.lmbd * loss_sup

        if cas_s is not None:
            # teacher student constrainte
            loss_st = self.l2_criterion(cas_s, cas_t)
            loss_total = loss_total + self.gamma * loss_st
            print("loss_st", (self.gamma*loss_st).detach().cpu().item())
            loss["loss_st"] = loss_st

        loss["loss_cls"] = loss_cls
        loss["loss_be"] = loss_be
        loss["loss_um"] = loss_um
        loss["loss_total"] = loss_total
        print("loss_cls", loss_cls.detach().cpu().item())
        print("loss_be", (self.beta * loss_be).detach().cpu().item())
        print("loss_um", (self.alpha * loss_um).detach().cpu().item())
        print("loss_total", loss_total.detach().cpu().item())

        return loss_total, loss

def train(net_student, net_teacher, loader_iter, optimizer, criterion, logger, step, m):
    net_student.train()
    
    _data, _label, _gt, _, _ = next(loader_iter)

    _data = _data.cuda()
    _label = _label.cuda()
    if _gt is not None:
        _gt = _gt.cuda()

    optimizer.zero_grad()

    score_act, score_bkg, feat_act, feat_bkg, _, cas_softmax_s, sup_cas_softmax = net_student(_data)
    _, _, _, _, _, cas_softmax_t, _ = net_student(_data)

    # cas = None
    # if net.self_train:
    #     feat_magnitudes_act = torch.mean(torch.norm(feat_act, dim=2), dim=1)
    #     feat_magnitudes_bkg = torch.mean(torch.norm(feat_bkg, dim=2), dim=1)
    #     feat_magnitudes = torch.norm(features, p=2, dim=2)
    #     feat_magnitudes = utils.minmax_norm(feat_magnitudes,
    #                                         max_val=feat_magnitudes_act,
    #                                         min_val=feat_magnitudes_bkg)
    #     feat_magnitudes = feat_magnitudes.repeat((cas_softmax.shape[-1], 1, 1)).permute(1, 2, 0)
    #     cas = utils.minmax_norm(cas_softmax * feat_magnitudes)

    # if step < 10:
    #     cas_softmax_s = None
    cost, loss = criterion(score_act, score_bkg, feat_act, feat_bkg, _label,
                           _gt, sup_cas_softmax, cas_softmax_s, cas_softmax_t)

    # update student parameters by backprapagation
    cost.backward()
    optimizer.step()
    # update teacher parameters by EMA
    student_params = OrderedDict(net_student.named_parameters())
    teacher_params = OrderedDict(net_teacher.named_parameters())

    # check if both model contains the same set of keys
    assert student_params.keys() == teacher_params.keys()

    for name, param in student_params.items():
        # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        # shadow_variable -= (1 - decay) * (shadow_variable - variable)
        teacher_params[name] = teacher_params[name] * m + (1 - m) * param

    student_buffers = OrderedDict(net_student.named_buffers())
    teacher_buffers = OrderedDict(net_teacher.named_buffers())

    # check if both model contains the same set of keys
    assert student_buffers.keys() == teacher_buffers.keys()

    for name, buffer in student_buffers.items():
        teacher_buffers[name] = teacher_buffers[name] * m + (1 - m) * buffer

    for key in loss.keys():
        logger.log_value(key, loss[key].cpu().item(), step)
