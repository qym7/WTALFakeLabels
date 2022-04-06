import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from itertools import product
import scipy.sparse as sp
import os
import sys
import random
import config


def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


def get_proposal_oic(tList, wtcam, final_score, c_pred, scale, v_len, sampling_frames, num_segments, _lambda=0.25, gamma=0.2):
    t_factor = (16 * v_len) / (scale * num_segments * sampling_frames)
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                if len(grouped_temp_list[j]) < 2:
                    continue

                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])

                len_proposal = len(grouped_temp_list[j])
                outer_s = max(0, int(grouped_temp_list[j][0] - _lambda * len_proposal))
                outer_e = min(int(wtcam.shape[0] - 1), int(grouped_temp_list[j][-1] + _lambda * len_proposal))

                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))

                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])

                c_score = inner_score - outer_score + gamma * final_score[c_pred[i]]
                t_start = grouped_temp_list[j][0] * t_factor
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([c_pred[i], c_score, t_start, t_end,
                               grouped_temp_list[j][0],
                               grouped_temp_list[j][-1] + 1])
            temp.append(c_temp)
    return temp


def result2json(result):
    result_file = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            line = {'label': config.class_dict[result[i][j][0]], 'score': result[i][j][1],
                    'segment': [result[i][j][2], result[i][j][3]],
                    'frame': [result[i][j][4], result[i][j][5]]}
            result_file.append(line)
    return result_file


def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def save_best_record_thumos(test_info, file_path, cls_thres, best_thres=None, map=False):
    fo = open(file_path, "w")
    fo.write("Step: {}\n".format(test_info["step"][-1]))
    if map:
        fo.write("Test_acc: {:.4f}\n".format(test_info["test_acc"][-1]))
        fo.write("average_mAP: {:.4f}\n".format(test_info["average_mAP"][-1]))
        tIoU_thresh = np.linspace(0.1, 0.7, 7)
        for i in range(len(tIoU_thresh)):
            fo.write("mAP@{:.1f}: {:.4f}\n".format(tIoU_thresh[i],
                    test_info["mAP@{:.1f}".format(tIoU_thresh[i])][-1]))

    # cls_thres = np.arange(0.1, 1, 0.1)
    fo.write("average_mIoU: {:.4f}\n".format(test_info["average_mIoU"][-1]))
    if best_thres is not None:
        fo.write("best_mIoU_thres: {:.2f}_{:.2f}\n".format(best_thres[0], best_thres[1]))
    for i in range(len(cls_thres)):
        fo.write("mIoU@{:.2f}_{:.2f}: {:.4f}\n".format(cls_thres[i][0],
                                                       cls_thres[i][1],
                 test_info["mIoU@{:.2f}_{:.2f}".format(cls_thres[i][0], cls_thres[i][1])][-1]))

    fo.write("average_precision: {:.4f}\n".format(test_info["average_precision"][-1]))
    if best_thres is not None:
        fo.write("best_precision_thres: {:.2f}_{:.2f}\n".format(best_thres[0], best_thres[1]))
    for i in range(len(cls_thres)):
        fo.write("precision@{:.2f}_{:.2f}: {:.4f}\n".format(cls_thres[i][0],
                                                       cls_thres[i][1],
                 test_info["precision@{:.2f}_{:.2f}".format(cls_thres[i][0], cls_thres[i][1])][-1]))

    fo.write("average_recall: {:.4f}\n".format(test_info["average_recall"][-1]))
    if best_thres is not None:
        fo.write("best_recall_thres: {:.2f}_{:.2f}\n".format(best_thres[0], best_thres[1]))
    for i in range(len(cls_thres)):
        fo.write("recall@{:.2f}_{:.2f}: {:.4f}\n".format(cls_thres[i][0],
                                                       cls_thres[i][1],
                 test_info["recall@{:.2f}_{:.2f}".format(cls_thres[i][0], cls_thres[i][1])][-1]))

    fo.write("average_f1score: {:.4f}\n".format(test_info["average_f1score"][-1]))
    if best_thres is not None:
        fo.write("best_f1score_thres: {:.2f}_{:.2f}\n".format(best_thres[0], best_thres[1]))
    for i in range(len(cls_thres)):
        fo.write("f1score@{:.2f}_{:.2f}: {:.4f}\n".format(cls_thres[i][0],
                                                       cls_thres[i][1],
                 test_info["f1score@{:.2f}_{:.2f}".format(cls_thres[i][0], cls_thres[i][1])][-1]))

    fo.close()



def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = nn.ReLU()
        max_val = relu(torch.max(act_map, dim=1)[0])
        min_val = relu(torch.min(act_map, dim=1)[0])
    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


def nms(proposals, thresh):
    proposals = np.array(proposals)
    x1 = proposals[:, 2]
    x2 = proposals[:, 3]
    scores = proposals[:, 1]

    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < thresh)[0]
        order = order[inds + 1]

    return keep


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def save_config(config, file_path):
    fo = open(file_path, "w")
    fo.write("Configurtaions:\n")
    fo.write(str(config))
    fo.close()

    
def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for
                        x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def get_cas(gt, cas):
    if gt.shape[0] > 750:
        samples = np.arange(gt.shape[0]) * 750 / gt.shape[0]
        samples = np.floor(samples)
        cas_ = cas[0][samples].cpu().numpy()
    else:
        cas_ = cas[0, :, :].cpu().numpy()
    return cas_

def calculate_iou(gt, pred_dict, cls_thres):
    iou = np.zeros(len(cls_thres))
    precision = np.zeros(len(cls_thres))
    recall = np.zeros(len(cls_thres))
    f1score = np.zeros(len(cls_thres))
    tp_lst = np.zeros(len(cls_thres))
    tn_lst = np.zeros(len(cls_thres))
    fp_lst = np.zeros(len(cls_thres))
    fn_lst = np.zeros(len(cls_thres))

    for j, thres in enumerate(cls_thres):
        bingo_count = 0
        total_count = 0
        tp = tn = fp = fn = 0
        for video_name in pred_dict:
            cas = pred_dict[video_name]
            gt_video = gt[video_name]
 
            for i in range(gt_video.shape[-1]):
                if sum(gt_video[:, i]) > 0:
                    gt_ = gt_video[:, i]
                    _thres = cls_thres[j][0] * cas[:, i].mean() + cls_thres[j][1]
                    pred = np.ones_like(cas[:, i]) * -1
                    pred[cas[:, i] <= _thres] = 0
                    pred[cas[:, i] > _thres] = 1

                    bingo_count += np.sum(gt_==pred)
                    total_count += len(pred)
                    tp_ = np.sum(np.logical_and(gt_==pred, gt_==1))
                    tn_ = np.sum(np.logical_and(gt_==pred, gt_==0))
                    tp += tp_
                    tn += tn_
                    fp += (np.sum(pred) - tp_)
                    fn += (np.sum(gt_) - tp_)

        iou[j] = bingo_count/total_count
        precision[j] = tp / (tp + fp + 1e-5)
        recall[j] = tp / (tp + fn + 1e-5)
        f1score[j] = 2*(precision[j]*recall[j])/(precision[j]+recall[j]+1e-5)
        tp_lst[j] = tp
        tn_lst[j] = tn
        fp_lst[j] = fp
        fn_lst[j] = fn
 
    return fp_lst, fn_lst, tp_lst, tn_lst
    # return iou, precision, recall, f1score


def encode_onehot(labels):
    classes = set(labels)
    class_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    
    label_onehot = np.array(list(map(class_dict.get, labels)),
                           dtype=np.int32)
    
    return label_onehot

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    
    return mx

def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
    np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)

def generate_adj_matrix(nodes_label):
    diff_edges = np.where(np.abs(np.diff(nodes_label)) == 1)[0]  # bkg和uncertain, act和uncertain之间的边
    diff_edges = [(i, i+1) for i in diff_edges]
    # l = len(nodes_label)
    # diff_edges = [(i, i+1) for i in range(l-1)]
    act_edges = np.where(nodes_label == 2)[0]  # act之间的边
    act_edges = list(product(act_edges, act_edges))
    bkg_edges = np.where(nodes_label == 0)[0]  # bkg之间的边
    bkg_edges = list(product(bkg_edges, bkg_edges))
    adj_cls = np.zeros((len(nodes_label), len(nodes_label)))
    adj_unc = np.zeros((len(nodes_label), len(nodes_label)))
    if len(diff_edges) > 0:
        np.add.at(adj_unc, tuple(zip(*diff_edges)), 1)
    np.add.at(adj_cls, tuple(zip(*act_edges)), 1)
    np.add.at(adj_cls, tuple(zip(*bkg_edges)), 1)
    np.fill_diagonal(adj_cls, 0)  # 消除act和bkg product中产生的自己指向自己的边，这个自指边在adjacent matrix后续normalize过程中会加上
    adj_cls = np.logical_or(adj_cls, (adj_cls.T)).astype(float)
    adj_unc = np.logical_or(adj_unc, (adj_unc.T)).astype(float)

    return adj_cls, adj_unc

def group_node(x, gt, thres1=0.2, thres2=0.4):
    '''
    gt: bs * T * 20
    return:
    nodes: list of bs elements, 每一个元素是Ni*2048的矩阵，表示N个同类视频中的节点
    '''
    x_label = np.ones_like(gt.detach().cpu().numpy())
    x_label[gt.detach().cpu().numpy()>thres2] = 2
    x_label[gt.detach().cpu().numpy()<=thres1] = 0
    nodes = []
    nodes_label = []
    nodes_pos = []
    vid_label = []
    for i, (feat, label) in enumerate(zip(x.detach().cpu().numpy(), x_label)):  # 迭代循环一类下的N个视频，由于每个视频产生的节点数不同，只能通过循环处理
        split_pos = np.where(np.diff(label) != 0)[0] + 1
        split_gt = np.split(label, split_pos)
        split_x = np.split(feat, split_pos)
        bg_pos = 0
        for j in range(len(split_pos)+1):
            nodes_label.append(split_gt[j].mean())
            node = split_x[j].mean(axis=0)
            nodes.append(node)
            vid_label.append(i)
            if j < len(split_pos):
                nodes_pos.append((i, bg_pos, bg_pos+len(split_x[j])))
                bg_pos += len(split_x[j])
            else:
                nodes_pos.append((i, bg_pos, gt.shape[-1]))
                bg_pos = gt.shape[-1]

    return np.stack(nodes), np.stack(nodes_label), nodes_pos, vid_label

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
