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


def save_best_record_thumos(test_info, file_path, cls_thres, best_thres=None):
    fo = open(file_path, "w")
    fo.write("Step: {}\n".format(test_info["step"][-1]))
    fo.write("Test_acc: {:.4f}\n".format(test_info["test_acc"][-1]))
    fo.write("average_mAP: {:.4f}\n".format(test_info["average_mAP"][-1]))

    tIoU_thresh = np.linspace(0.1, 0.7, 7)
    for i in range(len(tIoU_thresh)):
        fo.write("mAP@{:.1f}: {:.4f}\n".format(tIoU_thresh[i],
                 test_info["mAP@{:.1f}".format(tIoU_thresh[i])][-1]))

    # cls_thres = np.arange(0.1, 1, 0.1)
    fo.write("average_mIoU: {:.4f}\n".format(test_info["average_mIoU"][-1]))
    if best_thres is not None:
        fo.write("best_thres: {:.2f}_{:.2f}\n".format(best_thres[0], best_thres[1]))
    for i in range(len(cls_thres)):
        fo.write("mIoU@{:.2f}_{:.2f}: {:.4f}\n".format(cls_thres[i][0],
                                                       cls_thres[i][1],
                 test_info["mIoU@{:.2f}_{:.2f}".format(cls_thres[i][0], cls_thres[i][1])][-1]))

    fo.write("average_bkg_mIoU: {:.4f}\n".format(test_info["average_bkg_mIoU"][-1]))
    for i in range(len(cls_thres)):
        fo.write("bkg_mIoU@{:.2f}_{:.2f}: {:.4f}\n".format(cls_thres[i][0],
                                                           cls_thres[i][1],
                 test_info["bkg_mIoU@{:.2f}_{:.2f}".format(cls_thres[i][0], cls_thres[i][1])][-1]))

    fo.write("average_act_mIoU: {:.4f}\n".format(test_info["average_act_mIoU"][-1]))
    for i in range(len(cls_thres)):
        fo.write("act_mIoU@{:.2f}_{:.2f}: {:.4f}\n".format(cls_thres[i][0],
                                                           cls_thres[i][1],
                 test_info["act_mIoU@{:.2f}_{:.2f}".format(cls_thres[i][0], cls_thres[i][1])][-1]))

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
    test_iou = np.zeros(len(cls_thres))
    bkg_iou = np.zeros(len(cls_thres))
    act_iou = np.zeros(len(cls_thres))
    for video_name in pred_dict:
        cas = pred_dict[video_name]
        gt_video = gt[video_name]
        iou_ = np.zeros(len(cls_thres))
        bkg_iou_ = np.zeros(len(cls_thres))
        act_iou_ = np.zeros(len(cls_thres))
        for j, thres in enumerate(cls_thres):
            bingo_count = 0
            total_count = 0
            bkg_count = 0
            act_count = 0
            all_count = 0
            bkg_total_count = 0
            for i in range(gt_video.shape[-1]):
                if sum(gt_video[:, i]) > 0:
                    all_count += len(gt_video[:, i])
                    gt_ = gt_video[:, i]
                    
                    pred = np.ones_like(cas[:, i]) * -1
                    pred[cas[:, i] <= thres[0]] = 0
                    pred[cas[:, i] > thres[1]] = 1
                    gt_ = gt_video[:, i][pred >= 0]
                    pred = pred[pred >= 0]
                    
                    bingo_count += np.sum(gt_==pred)
                    # recall (gt_) / precision (pred)
                    bkg_count += np.sum(np.logical_and(gt_==pred,
                                                       gt_==0))
                    act_count += np.sum(np.logical_and(gt_==pred,
                                                       gt_==1))
                    total_count += len(pred)
                    bkg_total_count += np.sum(gt_==0)
            iou_[j] = bingo_count/total_count
            if bkg_total_count == 0:
                bkg_iou_[j] = 1
            else:
                bkg_iou_[j] = bkg_count/bkg_total_count
            if total_count-bkg_total_count == 0:
                act_iou_[j] = 0
            else:
                act_iou_[j] = act_count/(total_count-bkg_total_count)
            assert(act_count+bkg_count == bingo_count)
            # print(thres, round(total_count/all_count, 4),
            #       'iou', round(iou_[j], 4),
            #       'bkg_iou', round(bkg_iou_[j], 4),
            #       'act_iou', round(act_iou_[j], 4))
        test_iou += iou_
        bkg_iou += bkg_iou_
        act_iou += act_iou_

    test_iou = test_iou/len(pred_dict)
    bkg_iou = bkg_iou/len(pred_dict)
    act_iou = act_iou/len(pred_dict)
 
    return test_iou, bkg_iou, act_iou


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
    diff_edges = list(product(diff_edges, diff_edges+1))
    act_edges = np.where(nodes_label == 2)[0]  # act之间的边
    act_edges = list(product(act_edges, act_edges))
    bkg_edges = np.where(nodes_label == 2)[0]  # bkg之间的边
    bkg_edges = list(product(bkg_edges, bkg_edges))
    adj = np.zeros((len(nodes_label), len(nodes_label)))
    np.add.at(adj, tuple(zip(*diff_edges)), 1)
    np.add.at(adj, tuple(zip(*act_edges)), 1)
    np.add.at(adj, tuple(zip(*bkg_edges)), 1)
    np.fill_diagonal(adj, 0)  # 消除act和bkg product中产生的自己指向自己的边，这个自指边在adjacent matrix后续normalize过程中会加上
    adj = np.logical_or(adj, (adj.T)).astype(float)
    
    return adj