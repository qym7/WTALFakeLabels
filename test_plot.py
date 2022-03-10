import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import savgol_filter

import utils
import os
import json
import pickle
from eval.eval_detection import ANETdetection
from eval_utils import plot_pred, plot_node

supervision_path = '/GPFS/data/yimingqin/code/PTAL/outputs/LACP-n-5000/val_wtal_inner_pred_25.pickle'
ANNOT_PATH = '/GPFS/data/yimingqin/code/PTAL/dataset/thumos_annotations'
savefig_path='/GPFS/data/yimingqin/code/PTAL/outputs/LACP-n-5000/polynomial_filter'
save_path='/GPFS/data/yimingqin/code/PTAL/outputs/LACP-n-5000/'

if __name__ == "__main__":
    with open(os.path.join(ANNOT_PATH,
                           'val_gt_25.pickle'), 'rb') as f:
        gt = pickle.load(f)
    with open(os.path.join(ANNOT_PATH,
                           'val_label_dict.pickle'), 'rb') as f:
        label_dict = pickle.load(f)
    with open(supervision_path, 'rb') as f:
        pseudo_label = pickle.load(f)

    for vid_name in pseudo_label:
        gt_vid = gt[vid_name]
        to_draw = pseudo_label[vid_name]
        label = label_dict[vid_name]
        print(np.where(label==1)[0])
        for index in np.where(label==1)[0]:
            to_draw[:, index] = savgol_filter(to_draw[:, index], 17, 5, mode= 'nearest')
        # plot_pred(to_draw, gt_vid, 'polynomial-15-6' + vid_name, savefig_path)
    
    thres = np.arange(0.1, 1, 0.1)
    cls_thres = [(round(i, 2), round(i, 2)) for i in thres]
    test_iou, precision, recall, f1score = utils.calculate_iou(gt, pseudo_label, cls_thres)

    fo = open(os.path.join(save_path, '17-5-res.txt'), "w")
    fo.write("best iou: {}\n".format(max(test_iou)))
    fo.write("best f1score: {}\n".format(max(f1score)))
    for i in range(len(cls_thres)):
        fo.write("thres: {}\n".format(cls_thres[i]))
        fo.write("iou: {}\n".format(test_iou[i]))
        fo.write("f1score: {}\n".format(f1score[i]))
    fo.close()
