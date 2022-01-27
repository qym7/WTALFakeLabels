import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.sparse as sp

from utils import *


class CAS_Module(nn.Module):
    def __init__(self, len_feature, num_classes, self_train):
        super(CAS_Module, self).__init__()
        self.len_feature = len_feature
        self.self_train = self_train
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        # Dropout rate changing point, default 0.7
        self.drop_out = nn.Dropout(p=0.7)

        if self.self_train:
            self.sup_classifier = nn.Sequential(
                nn.Conv1d(in_channels=2048, out_channels=num_classes, kernel_size=1,
                        stride=1, padding=0, bias=False)
            )
            # Dropout rate changing point, default 0.7
            self.sup_drop_out = nn.Dropout(p=0.9)
            self.mlp = nn.Sequential(
                        nn.Linear(num_classes, num_classes),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(num_classes, num_classes),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(num_classes, num_classes)
            )

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv(out)
        features = out.permute(0, 2, 1)
        out = self.drop_out(out)
        out = self.classifier(out)
        out = out.permute(0, 2, 1)
        # out: (B, T, C)
        sup_out = None
        if self.self_train:
            sup_out = self.sup_drop_out(features.permute(0, 2, 1))
            sup_out = self.sup_classifier(sup_out)
            sup_out = sup_out.permute(0, 2, 1)
            # sup_out = self.mlp(sup_out)
            # sup_out = self.mlp(out)
            return out, features, sup_out
        return out, features, sup_out


class GraphConvolution(nn.Module):
    '''
    Reference: https://www.cnblogs.com/foghorn/p/15240260.html
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.drop_out = nn.Dropout(p=0.5)
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, input_features, adj):
        support = torch.mm(input_features, self.weight)  # 同weight相乘。
        # support = self.drop_out(support)
        # support = input_features # 
        output = torch.spmm(adj, support)  # 同adj mat相乘
        if self.use_bias:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    '''
    Reference: https://www.cnblogs.com/foghorn/p/15240260.html
    '''
    def __init__(self):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(2048, 2048)  # 如果保留feature长度不变的话，这个matrix有点大...
        self.gcn2 = GraphConvolution(2048, 2048)

    def get_adj(self, adj):
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        return adj

    def forward(self, X, adj):
        adj = self.get_adj(adj).cuda()
        X = F.relu(self.gcn1(X, adj))
        X = self.gcn2(X, adj)
        # X = self.gcn1(X, adj)
        # return F.log_softmax(X, dim=1)
        return X


class Model(nn.Module):
    def __init__(self, len_feature, num_classes, r_act, r_bkg,
                 self_train):
        super(Model, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.self_train = self_train

        self.GCN = GCN()
        self.cas_module = CAS_Module(len_feature, num_classes, self_train)

        self.softmax = nn.Softmax(dim=1)

        self.softmax_2 = nn.Softmax(dim=2)

        self.r_act = r_act
        self.r_bkg = r_bkg

        self.drop_out = nn.Dropout(p=0.7)
    
    def group_node(self, x, gt, thres1=0.2, thres2=0.4):
        '''
        gt: bs * T * 20
        return:
        nodes: list of bs elements, 每一个元素是Ni*2048的矩阵，表示N个同类视频中的节点
        '''
        x_label = np.ones_like(gt.detach().cpu().numpy()) * 1
        x_label[gt.detach().cpu().numpy()>thres2] = 2
        x_label[gt.detach().cpu().numpy()<=thres1] = 0
        nodes = []
        nodes_label = []
        nodes_pos = []
        for i, (feat, gt_vid) in enumerate(zip(x.detach().cpu().numpy(), x_label)):  # 迭代循环一类下的N个视频，由于每个视频产生的节点数不同，只能通过循环处理
            split_pos = np.where(np.diff(gt_vid) != 0)[0] + 1
            split_gt = np.split(gt_vid, split_pos)
            split_x = np.split(feat, split_pos)
            bg_pos = 0
            for j in range(len(split_pos)+1):
                nodes_label.append(split_gt[j].mean())
                node = split_x[j].mean(axis=0)
                nodes.append(node)
                if j < len(split_pos):
                    nodes_pos.append((i, bg_pos, bg_pos+len(split_x[j])))
                    bg_pos += len(split_x[j])
                else:
                    nodes_pos.append((i, bg_pos, 750))
                    bg_pos = 750

        return np.stack(nodes), np.stack(nodes_label), nodes_pos

    def forward(self, x, gt=None, index=None):
        nodes = []
        nodes_label = []

        if gt is not None:
            if index is None:
                raise('index should not be none!')
            for i in range(len(x)):  # bs * N * T * 20, bs也意味着有bs类的视频，每类视频有N个
                vid_cls = x[i]
                gt_cls = gt[i, :, :, index[i]]  # N * T
                nodes_, nodes_label_, nodes_pos_ = self.group_node(vid_cls, gt_cls)  # 由于只有在同类视频里产生节点图，所以需要迭代循环所有类
                nodes.append(torch.from_numpy(nodes_).cuda())  # 产生N个同类视频的节点
                nodes_label.append(torch.from_numpy(nodes_label_).cuda())  # 产生上述节点对应标签，1为action，0为bkg，-1为不确定
                adj = generate_adj_matrix(nodes_label_)
                # pass to GCN
                x_ = self.GCN(torch.from_numpy(nodes_).cuda(), adj)  # 根据gcn处理后的node和node在特征中的位置更新features
                for j, pos in enumerate(nodes_pos_):
                    x[i, pos[0], pos[1]:pos[2]] = x_[j].repeat(pos[2]-pos[1], 1)

            gt = gt.reshape(-1, gt.shape[-2], gt.shape[-1])

        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        num_segments = x.shape[1]
        k_act = num_segments // self.r_act
        k_bkg = num_segments // self.r_bkg

        cas, features, sup_cas = self.cas_module(x)
        sup_cas_softmax = None
        if self.self_train:
            # sup_cas_softmax = self.softmax_2(sup_cas)  # need to use sigmoid after
            sup_cas_softmax = sup_cas

        feat_magnitudes = torch.norm(features, p=2, dim=2)

        select_idx = torch.ones_like(feat_magnitudes).cuda()
        select_idx = self.drop_out(select_idx)

        feat_magnitudes_drop = feat_magnitudes * select_idx

        feat_magnitudes_rev = torch.max(feat_magnitudes, dim=1, keepdim=True)[0] - feat_magnitudes
        feat_magnitudes_rev_drop = feat_magnitudes_rev * select_idx

        _, sorted_idx = feat_magnitudes_drop.sort(descending=True, dim=1)
        idx_act = sorted_idx[:, :k_act]
        idx_act_feat = idx_act.unsqueeze(2).expand([-1, -1, features.shape[2]])

        _, sorted_idx = feat_magnitudes_rev_drop.sort(descending=True, dim=1)
        idx_bkg = sorted_idx[:, :k_bkg]
        idx_bkg_feat = idx_bkg.unsqueeze(2).expand([-1, -1, features.shape[2]])
        idx_bkg_cas = idx_bkg.unsqueeze(2).expand([-1, -1, cas.shape[2]])
        
        feat_act = torch.gather(features, 1, idx_act_feat)
        feat_bkg = torch.gather(features, 1, idx_bkg_feat)

        sorted_scores, _= cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k_act, :]
        score_act = torch.mean(topk_scores, dim=1)
        score_bkg = torch.mean(torch.gather(cas, 1, idx_bkg_cas), dim=1)

        score_act = self.softmax(score_act)
        score_bkg = self.softmax(score_bkg)

        cas_softmax = self.softmax_2(cas)

        return score_act, score_bkg, feat_act, feat_bkg, features, cas_softmax, sup_cas_softmax, nodes, nodes_label
