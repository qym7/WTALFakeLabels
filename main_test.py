'''
Author: your name
Date: 2021-12-18 19:08:21
LastEditTime: 2021-12-28 14:10:41
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /yimingqin/code/WTAL-Uncertainty-Modeling/main_eval.py
'''
import pickle
import pdb
import numpy as np
import torch.utils.data as data
import utils
from options import *
from config import *
from test import *
from model import *
from tensorboard_logger import Logger
from thumos_features import *

from eval_utils import ANNOT_PATH


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    config = Config(args)
    worker_init_fn = None

    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)

    if config.test_dataset == 'test':
        utils.save_config(config, os.path.join(config.output_path, "config.txt"))

    net = Model(config.len_feature, config.num_classes, config.r_act, config.r_bkg,
                False)
    net = net.cuda()

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode=config.test_dataset,
                        modal=config.modal, feature_fps=config.feature_fps,
                        num_segments=config.num_segments, supervision='weak',
                        seed=config.seed, sampling='uniform'),
            batch_size=1,
            shuffle=False, num_workers=config.num_workers,
            worker_init_fn=worker_init_fn)
    with open(os.path.join(ANNOT_PATH,
                           '{}_gt_25.pickle'.format(config.test_dataset)), 'rb') as f:
        gt = pickle.load(f)

    cls_thres = np.arange(0.1, 1, 0.1)
    test_info = {"step": [], "test_acc": [],
                 "average_mAP": [],
                 "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [], 
                 "mAP@0.4": [], "mAP@0.5": [], "mAP@0.6": [], "mAP@0.7": [],
                 "average_mIoU": []}
    iou_info = {'mIoU@{:.2f}'.format(thres): [] for thres in cls_thres}
    test_info.update(iou_info)

    logger = Logger(config.log_path)

    test(net, config, logger, test_loader, test_info, 0, gt,
         cls_thres=cls_thres, model_file=config.model_file,
         save=config.save)

    utils.save_best_record_thumos(test_info, 
        os.path.join(config.output_path, "test_record_{}.txt".format(config.test_dataset)),
        cls_thres=cls_thres)
