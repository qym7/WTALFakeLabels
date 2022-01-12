import torch
import torch.nn as nn
import numpy as np

import utils


class UM_loss(nn.Module):
    def __init__(self, alpha, beta, lmbd, bkg_lmbd, margin, thres):
        super(UM_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lmbd = lmbd
        self.bkg_lmbd = bkg_lmbd
        self.margin = margin
        self.thres = thres
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

    def balanced_ce(self, gt, cas, gt_class):
        act_loss = torch.tensor(0.).cuda()
        bkg_loss = torch.tensor(0.).cuda()
        act_count = 0
        bkg_count = 0
        gt = (gt > self.thres).float().cuda() 
        for i in range(cas.shape[0]):
            for j in range(cas.shape[-1]):
                if gt_class[i, j] > 0:
                    _loss = self.BCE(gt[i, :, j], cas[i, :, j])
                    # _loss = torch.norm(cas[i, :, j] - gt[i, :, j], p=2)
                    act_loss = act_loss + torch.mean(_loss)
                    act_count += 1
                else:
                    if self.bkg_lmbd > 0:
                        _gt = torch.zeros_like(cas[i, :, j]).cuda()
                        _loss = self.BCE(_gt, cas[i, :, j])
                        # _loss = torch.norm(cas[i, :, j] - _gt, p=2)
                        bkg_loss = bkg_loss + torch.mean(_loss)
                    bkg_count += 1
        act_loss = act_loss / act_count
        bkg_loss = bkg_loss / bkg_count

        return act_loss, bkg_loss

    def forward(self, score_act, score_bkg, feat_act, feat_bkg, label,
                gt, cas):
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
            loss_sup_act, loss_sup_bkg = self.balanced_ce(gt, cas, label)
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

        loss["loss_cls"] = loss_cls
        loss["loss_be"] = loss_be
        loss["loss_um"] = loss_um
        loss["loss_total"] = loss_total
        print("loss_cls", loss_cls.detach().cpu().item())
        print("loss_be", (self.beta * loss_be).detach().cpu().item())
        print("loss_um", (self.alpha * loss_um).detach().cpu().item())
        print("loss_total", loss_total.detach().cpu().item())

        return loss_total, loss

def train(net, loader_iter, optimizer, criterion, logger, step):
    net.train()

    _data, _label, _gt, _, _ = next(loader_iter)

    _data = _data.cuda()
    _label = _label.cuda()
    if _gt is not None:
        _gt = _gt.cuda()

    optimizer.zero_grad()

    score_act, score_bkg, feat_act, feat_bkg, _, _, sup_cas_softmax = net(_data)

    cost, loss = criterion(score_act, score_bkg, feat_act, feat_bkg, _label,
                           _gt, sup_cas_softmax)

    cost.backward()
    optimizer.step()

    for key in loss.keys():
        logger.log_value(key, loss[key].cpu().item(), step)
