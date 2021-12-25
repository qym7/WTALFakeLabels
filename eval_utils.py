import os
import json
import pickle

from matplotlib import colors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import class_dict

ANNOT_PATH = '/GPFS/data/yimingqin/datasets/thumos_annotations'
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

if __name__ == '__main__':
    save_gt()