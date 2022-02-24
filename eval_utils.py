import os
import json
import pickle

from matplotlib import colors
from scipy.signal import savgol_filter

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import class_dict

ANNOT_PATH = '/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/dataset/thumos_annotations'
DATA_PATH = '/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14/features/val/flow'
DATA_TEST_PATH = '/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14/features/test/flow'

cls_dict = {7: 0,
            9: 1,
            12: 2,
            21: 3,
            22: 4,
            23: 5,
            24: 6,
            26: 7,
            31: 8,
            33: 9,
            36: 10,
            40: 11,
            45: 12,
            51: 13,
            68: 14,
            79: 15,
            85: 16,
            92: 17,
            93: 18,
            97: 19}


def reverse_dict(dict):
    new_dict = {}
    for i in dict:
        new_dict[dict[i]] = i
    return new_dict


def save_gt():
    '''
    Function to save action classification ground truth with following rules:
        - Ground Truth: N_videos * T * (N_classes + 1)
    where 
        - the 1st dimension of (N_classes + 1) means if it is background or not,
        - fps = 25
    '''
    val_path = os.path.join(ANNOT_PATH, 'val_Annotation.csv')
    test_path = os.path.join(ANNOT_PATH, 'test_Annotation.csv')
    val_info_path = os.path.join(ANNOT_PATH, 'val_video_info.csv')
    test_info_path = os.path.join(ANNOT_PATH, 'test_video_info.csv')
    val = pd.read_csv(val_path).groupby('video')
    val_info = pd.read_csv(val_info_path).set_index('video')
    test = pd.read_csv(test_path).groupby('video')
    test_info = pd.read_csv(test_info_path).set_index('video')

    val_dict = {}
    test_dict = {}
    for row in val:
        if row[0] in val_info.index:
            feat = np.load(os.path.join(DATA_PATH, row[0]+'.npy'))
            len_v = feat.shape[0]
            # len_v = int(int(float(val_info.loc[row[0]]['count'])*25/30)/16)
            # print(row[0], len_v, float(val_info.loc[row[0]]['count'])*25/30/16)
            # if len_v > 750:
            #     val_dict[row[0]] = np.zeros((750, 20))
            #     for r in row[1].iterrows():
            #         if r[1]['type_idx'] != 0:
            #             val_dict[row[0]][int(r[1]['start']*25/16*750/len_v):
            #                               int(r[1]['end']*25/16*750/len_v + 1),
            #                               int(cls_dict[r[1]['type_idx']])] = 1
            # else:
            val_dict[row[0]] = np.zeros((len_v, 20))
            for r in row[1].iterrows():
                if r[1]['type_idx'] != 0:
                    val_dict[row[0]][int(r[1]['start']*25/16):
                                    int(r[1]['end']*25/16 + 1),
                                    int(cls_dict[r[1]['type_idx']])] = 1
    for row in test:
        if row[0] in test_info.index:
            feat = np.load(os.path.join(DATA_TEST_PATH, row[0]+'.npy'))
            len_v = feat.shape[0]
            # len_v = int(int(float(test_info.loc[row[0]]['count'])*25/30)/16)
            # if len_v > 750:
            #     test_dict[row[0]] = np.zeros((750, 20))
            #     len_v = int(test_info.loc[row[0]]['count']*25/30/16)
            #     for r in row[1].iterrows():
            #         if r[1]['type_idx'] != 0:
            #             test_dict[row[0]][int(r[1]['start']*25/16*750/len_v):
            #                               int(r[1]['end']*25/16*750/len_v + 1),
            #                               int(cls_dict[r[1]['type_idx']])] = 1
            # else:
            test_dict[row[0]] = np.zeros((len_v, 20))
            for r in row[1].iterrows():
                if r[1]['type_idx'] != 0:
                    test_dict[row[0]][int(r[1]['start']*25/16):
                                    int(r[1]['end']*25/16 + 1),
                                    int(cls_dict[r[1]['type_idx']])] = 1
    file_to_write = open(os.path.join(ANNOT_PATH, 'val_gt_25.pickle'), 'wb')
    pickle.dump(val_dict, file_to_write)
    file_to_write = open(os.path.join(ANNOT_PATH, 'test_gt_25.pickle'), 'wb')
    pickle.dump(test_dict, file_to_write)
    
    # gt_test = {}
    # gt_test['database'] = {}
    # for row in test:
    #     if row[0] in test_info.index:
    #         gt_test['database'][row[0]] = {}
    #         gt_test['database'][row[0]]['annotations'] = []
    #         for r in row[1].iterrows():
    #             if r[1]['type_idx'] != 0:
    #                 gt_test['database'][row[0]]['annotations'].append({
    #                     'label': r[1]['type'],
    #                     'segment': [float(r[1]['start']), float(r[1]['end'])]
    #                 })
    # with open(os.path.join(ANNOT_PATH, 'gt_test.json'), 'w') as outfile:
    #     json.dump(gt_test, outfile)

def turn_time2frame(file_folder, thres):
    name_dict = reverse_dict(class_dict)
    file_path = os.path.join(file_folder, 'result.json')
    with open(file_path, 'rb') as f:
        res_time = json.load(f)['results']
    res_frame = {}
    for vid_name in res_time:
        feat = np.load(os.path.join(DATA_PATH, vid_name+'.npy'))
        len_v = feat.shape[0]
        res_ = np.zeros((len_v, 20))
        for i in range(len(res_time[vid_name])):
            annot_ = res_time[vid_name][i]
            if annot_['score'] > thres:
                str_ = annot_['segment'][0]
                end_ = annot_['segment'][1]
                res_[int(str_*25/16):
                    int(end_*25/16 + 1),
                    int(name_dict[annot_['label']])] = 1

        res_frame[vid_name] = res_

    # file_to_write = open(os.path.join(file_folder, 'Wval_pred_25.pickle'), 'wb')
    # pickle.dump(res_frame, file_to_write)
    return res_frame

def test_iou(file_folder):
    with open(os.path.join(ANNOT_PATH, 'val_gt_25.pickle'), 'rb') as f:
        gt = pickle.load(f)
    score_thres = np.arange(0, 1, 0.01)
    test_iou = np.zeros(len(score_thres))
    for j, thres in enumerate(score_thres):
        wpred = turn_time2frame(file_folder, thres)
        iou_ = 0
        for video_name in wpred:
            cas = wpred[video_name]
            gt_video = gt[video_name]
            bingo_count = 0
            total_count = 0
            for i in range(gt_video.shape[-1]):
                if sum(gt_video[:, i]) > 0:
                    pred = cas[:, i] > thres
                    bingo_count += np.sum(gt_video[:, i]==pred)
                    total_count += len(pred)
            iou_ += bingo_count/total_count
        test_iou[j] = iou_/len(gt)
        print(thres, test_iou[j])


def save_pred():
    '''
    Function to save action classification prediction with following rules:
        - Prediction: N_videos * T * (N_classes + 1)
    where 
        - the 1st dimension of (N_classes + 1) means if it is background or not,
        - fps = 25
    '''
    pass


def plot_one_class(cas, gt, class_name, video_name, save_path):
    '''
    Function to plot and save
    Args:
        - cas: 20
        - gt: 20
    '''
    plt.figure()
    plt.plot(range(cas.shape[0]), cas, linewidth=1, color='b', label='pred')
    gt = list(gt)
    x = list(range(cas.shape[0]))
    i = 0
    while i < len(gt) - 1:
        if gt[i+1] - gt[i] > 0:
            gt = gt[:i+1] + [0] + gt[i+1:]
            x = x[:i+1] + [x[i+1]] + x[i+1:]
            i += 2
        elif gt[i+1] - gt[i] < 0:
            gt = gt[:i+1] + [0] + gt[i+1:]
            x = x[:i+1] + [x[i]] + x[i+1:]
            i += 2
        else:
            i += 1
    # print(gt, x)
    # plt.plot(range(cas.shape[0]), gt, linewidth=1, color='r', label='gt')
    plt.plot(x, gt, linewidth=1, color='r', label='gt')
    plt.xlabel('T')
    plt.ylabel('Probability')
    plt.title('{} - {}'.format(video_name, class_name))
    plt.legend()
    plt.savefig(os.path.join(save_path, '{}_{}.png'.format(video_name, class_name)))


def plot_pred(cas, gt, video_name, save_path):
    '''
    Function to plot and save
    Args:
        - cas: T * 20
        - gt: T * 20
    '''
    for i in range(cas.shape[1]):
        if sum(gt[:, i]) > 0:
            class_name = class_dict[i]
            plot_one_class(cas[:, i], gt[:, i], class_name,
                           video_name, save_path)


def visualise_PCA(X, y, class_lst, save_path):
    pca_2 = PCA(n_components=2)
    pca_2_res = pca_2.fit_transform(X)
    pca_3 = PCA(n_components=3)
    pca_3_res = pca_3.fit_transform(X)

    plt.figure(figsize=(5, 5))
    act_pos = y == 2
    bkg_pos = y == 0
    unk_pos = y == 1

    plt.scatter(pca_2_res[:, 0][act_pos], pca_2_res[:, 1][act_pos],
                c=class_lst[act_pos], label="PCA", cmap='Oranges')
    plt.colorbar()
    plt.scatter(pca_2_res[:, 0][bkg_pos], pca_2_res[:, 1][bkg_pos],
                c=class_lst[bkg_pos], label="PCA", cmap='Greens')
    plt.colorbar()
    plt.scatter(pca_2_res[:, 0][unk_pos], pca_2_res[:, 1][unk_pos],
                c=class_lst[unk_pos], label="PCA", cmap='Blues')
    plt.colorbar()
    plt.xlabel(pca_2.explained_variance_ratio_[0])
    plt.ylabel(pca_2.explained_variance_ratio_[1])
    plt.title('PCA2 for nodes of different type')
    plt.legend()
    file_path = os.path.join(save_path, 'inner_pca2.png')
    plt.savefig(file_path, dpi=120)


def plot_node(nodes, nodes_label, class_lst, save_path):
    '''
    Function to plot and save
    Args:
        - nodes: list of numpy array of size Ni * 2048
        - nodes_label: list of numpy array of size Ni
        - label: list of ints, class of one element in nodes
    '''
    nodes = np.concatenate(nodes)
    nodes_label = np.concatenate(nodes_label)
    class_lst = np.concatenate(class_lst)
    num_act = len(nodes)
    visualise_PCA(nodes,
                  nodes_label,
                  class_lst,
                  save_path)
    # for i in range(cas.shape[1]):
    #     if sum(gt[:, i]) > 0:
    #         class_name = class_dict[i]
    #         plot_one_class(cas[:, i], gt[:, i], class_name,
    #                        video_name, save_path)

if __name__ == '__main__':
    # save_gt()
    # turn_time2frame('/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/outputs/UM')
    test_iou('/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/outputs/UM')