'''
Author: your name
Date: 2021-12-16 17:00:30
LastEditTime: 2021-12-23 15:46:01
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/main.py
'''
import pickle
import pdb
import numpy as np
import torch.utils.data as data
import utils
from options import *
from config import *
from train import *
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

    utils.save_config(config, os.path.join(config.output_path, "config.txt"))

    net = Model(config.len_feature, config.num_classes, config.r_act, config.r_bkg)
    net = net.cuda()

    train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                        modal=config.modal, feature_fps=config.feature_fps,
                        num_segments=config.num_segments, supervision='weak',
                        seed=config.seed, sampling='random'),
            batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers,
            worker_init_fn=worker_init_fn)

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                        modal=config.modal, feature_fps=config.feature_fps,
                        num_segments=config.num_segments, supervision='weak',
                        seed=config.seed, sampling='uniform'),
            batch_size=1,
            shuffle=False, num_workers=config.num_workers,
            worker_init_fn=worker_init_fn)

    with open(os.path.join(ANNOT_PATH, 'test_gt_25.pickle'), 'rb') as f:
        gt = pickle.load(f)

    cls_thres = np.arange(0.1, 1, 0.1)
    test_info = {"step": [], "test_acc": [],
                 "average_mAP": [],
                 "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [], 
                 "mAP@0.4": [], "mAP@0.5": [], "mAP@0.6": [], "mAP@0.7": [],
                 "average_mIoU": []}
    iou_info = {'mIoU@{:.2f}'.format(thres): [] for thres in cls_thres}
    test_info.update(iou_info)
    cls_thres = np.arange(0.1, 1, 0.1)
    best_mIoU = -1
    best_thres = 0

    criterion = UM_loss(config.alpha, config.beta, config.margin)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr[0],
        betas=(0.9, 0.999), weight_decay=0.0005)

    logger = Logger(config.log_path)

    for step in tqdm(
            range(1, config.num_iters + 1),
            total = config.num_iters,
            dynamic_ncols = True
        ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_loader) == 0:
            loader_iter = iter(train_loader)

        train(net, loader_iter, optimizer, criterion, logger, step)

        test(net, config, logger, test_loader, test_info, step, gt,
             cls_thres=cls_thres, datatype='test')
        iou = [test_info['mIoU@{:.2f}'.format(thres)][-1] for thres in cls_thres]

        if max(iou) > best_mIoU:
            best_mIoU = max(iou)
            best_thres = cls_thres[np.argmax(np.array(iou))]

            utils.save_best_record_thumos(test_info,
                os.path.join(config.output_path, "best_record_thres_{}_seed_{}.txt".format(
                    best_thres, config.seed)))

            torch.save(net.state_dict(), os.path.join(args.model_path, \
                "model_seed_{}_{}.pkl".format(best_thres, config.seed)))

        logger.log_value('Best mIoU threshold', best_thres, step)
        logger.log_value('Best mIoU', best_mIoU, step)

        print('--Average mIoU ', round(test_info['average_mIoU'][-1], 4),
              '--Best mIoU ', round(best_mIoU, 4),
              '--Best mIoU Thres ', round(best_thres, 4))

    utils.save_best_record_thumos(test_info,
        os.path.join(config.output_path, "best_record_seed_{}.txt".format(config.seed)),
        cls_thres=cls_thres)
