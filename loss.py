import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class GCNN_loss(nn.Module):
    def __init__(self, gcnn_weight):
        super(GCNN_loss, self).__init__()
        self.gcnn_weight = gcnn_weight
        self.loss = ContrastiveLoss(0.5)

    def construct_pairs(self, nodes, nodes_label, nodes_nbr):
        # ignore the gradient caused by following instructions
        nodes = torch.cat(nodes)
        with torch.no_grad():
            # merge nodes
            for i in range(len(nodes_label)):
                labels = nodes_label[i]
                labels[labels==2] = labels[labels==2] + i 
                nodes_label[i] = labels
            node_mask = (nodes_label!=1) & (nodes_label>=0)
            nodes_label = torch.cat([nodes_label[i][node_mask[i]]
                                       for i in range(len(nodes_label))])
            # find pairs
            similarity_matrix = utils.sim_matrix(nodes, nodes)
            # mask all pairs of different classes
            mask = nodes_label.unsqueeze(1) - nodes_label.unsqueeze(0)
            mask = mask == 0
            zero_similarity_matrix = similarity_matrix.clone()
            # eliminate nodes of different type
            zero_similarity_matrix[~mask] = 0
            zero_similarity_matrix.fill_diagonal_(0)

        pair_nodes = nodes[zero_similarity_matrix.argmax(dim=0)]

        return nodes, pair_nodes, nodes_label, mask

    def forward(self, nodes, nodes_label, nodes_nbr):
        nodes, pair_nodes, nodes_label, mask = self.construct_pairs(nodes, nodes_label, nodes_nbr)
        loss = self.loss(nodes, pair_nodes, nodes_label, mask)

        return loss * self.gcnn_weight


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, nodes, pair_nodes, nodes_labels, mask):

        z = F.normalize(nodes, dim=1)
        z_p = F.normalize(pair_nodes, dim=1)
        positive_similarity = F.cosine_similarity(z.unsqueeze(1), z_p.unsqueeze(0), dim=2)

        nominator = torch.exp(positive_similarity / self.temperature)

        similarity_matrix = utils.sim_matrix(nodes, nodes)
        # eliminate nodes of same type
        denominator = (~mask).int() * torch.exp(similarity_matrix / self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / nodes.shape[0]

        return loss


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

        loss["loss_cls"] = loss_cls
        loss["loss_be"] = loss_be
        loss["loss_um"] = loss_um
        loss["loss_total"] = loss_total
        print("loss_cls", loss_cls.detach().cpu().item())
        print("loss_be", (self.beta * loss_be).detach().cpu().item())
        print("loss_um", (self.alpha * loss_um).detach().cpu().item())
        print("loss_total", loss_total.detach().cpu().item())

        return loss_total, loss
