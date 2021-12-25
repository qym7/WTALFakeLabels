'''
Author: your name
Date: 2021-12-23 09:45:42
LastEditTime: 2021-12-25 23:24:24
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /yimingqin/code/WTAL-Uncertainty-Modeling/config.py
'''
import numpy as np
import os

class Config(object):
    def __init__(self, args):
        self.lr = eval(args.lr)
        self.lr_str = args.lr
        self.num_iters = len(self.lr)
        self.num_classes = 20
        self.modal = args.modal
        if self.modal == 'all':
            self.len_feature = 2048
        else:
            self.len_feature = 1024
        self.batch_size = args.batch_size
        self.data_path = args.data_path
        self.save = int(args.save)
        self.model_path = args.model_path
        self.output_path = args.output_path
        self.log_path = args.log_path
        self.num_workers = args.num_workers
        self.alpha = args.alpha
        self.beta = args.beta
        self.margin = args.margin
        self.r_act = args.r_act
        self.r_bkg = args.r_bkg
        self.class_thresh = args.class_th
        self.act_thresh_cas = np.arange(0.0, 0.25, 0.025)
        self.act_thresh_magnitudes = np.arange(0.4, 0.625, 0.025)
        self.scale = 24
        self.gt_path = os.path.join(self.data_path, 'gt.json')
        # self.f_gt_path = os.path.join(self.script_path, 'val_gt_25.pickle')
        self.model_file = args.model_file
        self.seed = args.seed
        self.feature_fps = 25
        self.num_segments = 750
        # new args
        self.test_dataset = args.test_dataset
        self.supervision = args.supervision
        self.supervision_path = args.supervision_path
        self.thres = args.thres
        self.a1 = args.a1
        self.a2 = args.a2
        self.save = bool(args.save)
        self.m = args.m
        self.gamma = args.gamma

    def __str__(self):
        attrs = vars(self)
        attr_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attr_lst if item != 'lr')


class_dict = {0: 'BaseballPitch',
                1: 'BasketballDunk',
                2: 'Billiards',
                3: 'CleanAndJerk',
                4: 'CliffDiving',
                5: 'CricketBowling',
                6: 'CricketShot',
                7: 'Diving',
                8: 'FrisbeeCatch',
                9: 'GolfSwing',
                10: 'HammerThrow',
                11: 'HighJump',
                12: 'JavelinThrow',
                13: 'LongJump',
                14: 'PoleVault',
                15: 'Shotput',
                16: 'SoccerPenalty',
                17: 'TennisSwing',
                18: 'ThrowDiscus',
                19: 'VolleyballSpiking'}