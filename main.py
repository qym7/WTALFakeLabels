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
    
    # =================== booking a gpu ==================
    # careful, you have to first load the data
    print('current device: ', torch.cuda.current_device())
    free_gpu_id = int(utils.get_free_gpu())
    print('free gpu: ', free_gpu_id)
    torch.cuda.set_device(free_gpu_id)
    print('current device: ', torch.cuda.current_device())

    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)

    utils.save_config(config, os.path.join(config.output_path, "config.txt"))

    net = Model(config.len_feature, config.num_classes, 
                config.r_act, config.r_bkg,
                config.supervision!='weak')
    net = net.cuda()

    train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                        modal=config.modal, feature_fps=config.feature_fps,
                        num_segments=config.num_segments, supervision=config.supervision,
                        supervision_path=config.supervision_path,
                        seed=config.seed, sampling='random'),
            batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers,
            worker_init_fn=worker_init_fn)

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode=config.test_dataset,
                        modal=config.modal, feature_fps=config.feature_fps,
                        num_segments=config.num_segments, supervision='weak',
                        seed=config.seed, sampling='uniform'),
            batch_size=1,
            shuffle=False, num_workers=config.num_workers,
            worker_init_fn=worker_init_fn)

    with open(os.path.join(ANNOT_PATH, '{}_gt_25.pickle'.format(config.test_dataset)), 'rb') as f:
        gt = pickle.load(f)

    cls_thres = np.arange(0.1, 1, 0.1)
    test_info = {"step": [], "test_acc": [],
                 "average_mAP": [],
                 "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [], 
                 "mAP@0.4": [], "mAP@0.5": [], "mAP@0.6": [], "mAP@0.7": [],
                 "average_mIoU": [], "average_bkg_mIoU": [], "average_act_mIoU": []}
    iou_info = {'mIoU@{:.2f}'.format(thres): [] for thres in cls_thres}
    iou_info.update({'bkg_mIoU@{:.2f}'.format(thres): [] for thres in cls_thres})
    iou_info.update({'act_mIoU@{:.2f}'.format(thres): [] for thres in cls_thres})
    test_info.update(iou_info)
    best_mAP = -1
    cls_thres = np.arange(0.1, 1, 0.1)
    best_mIoU = best_bkg_mIoU = best_act_mIoU = -1
    best_thres = 0

    criterion = UM_loss(config.alpha, config.beta, config.lmbd,
                        config.margin, config.thres)

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
             cls_thres=cls_thres, save=False)

        iou = [test_info['mIoU@{:.2f}'.format(thres)][-1] for thres in cls_thres]

        # save model by mIoU
        if max(iou) > best_mIoU:
            iou_idx = np.argmax(np.array(iou))
            best_mIoU = max(iou)
            best_thres = cls_thres[iou_idx]
            best_bkg_mIoU = test_info['bkg_mIoU@{:.2f}'.format(best_thres)][0]
            best_act_mIoU = test_info['act_mIoU@{:.2f}'.format(best_thres)][0]

            utils.save_best_record_thumos(test_info,
                os.path.join(config.output_path, "best_record_{}_seed_{}.txt".format(
                    config.test_dataset,
                    config.seed)),
                    cls_thres=cls_thres,
                    best_thres=best_thres)

            torch.save(net.state_dict(), os.path.join(args.model_path, \
                "model_seed_{}.pkl".format(config.seed)))

        logger.log_value('Best mIoU threshold', best_thres, step)
        logger.log_value('Best mIoU', best_mIoU, step)

        print(config.model_path.split('/')[-1],
              '--Average mIoU ', round(test_info['average_mIoU'][-1], 4),
              '--Best mIoU ', round(best_mIoU, 4),
              '--Best mIoU Thres ', round(best_thres, 4),
              '--Bkg mIoU ', round(best_bkg_mIoU, 4),
              '--Act mIoU ', round(best_act_mIoU, 4))
        
        # # save model by mAP
        # if test_info["average_mAP"][-1] > best_mAP:
        #     best_mAP = test_info["average_mAP"][-1]

        #     utils.save_best_record_thumos(test_info, 
        #         os.path.join(config.output_path, "best_record_{}_seed_{}.txt".format(
        #             config.test_dataset,
        #             config.seed)),
        #         cls_thres=cls_thres)

        #     torch.save(net.state_dict(), os.path.join(args.model_path, \
        #         "model_seed_{}.pkl".format(config.seed)))

        print(config.model_path.split('/')[-1],
              'mAP', round(test_info["average_mAP"][-1], 4))
            #   'mIoU', test_info["average_mIoU"][-1])

    utils.save_best_record_thumos(test_info,
        os.path.join(config.output_path, "full_record_{}_seed_{}.txt".format(config.test_dataset,
                                                                             config.seed)),
        cls_thres=cls_thres,
        best_thres=best_thres)
