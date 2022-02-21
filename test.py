import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
import os
import json
import pickle
from eval.eval_detection import ANETdetection
from eval_utils import plot_pred, plot_node


def test(net, gcnn, config, logger, test_loader, test_info, step, gt,
         cls_thres=np.arange(0.1, 1, 0.1),
         model_file=None, save=False, mAP=False):
    with torch.no_grad():
        net.eval()

        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        wtal_pred_dict = {}
        sup_pred_dict = {}
        nodes_lst = []
        nodes_label_lst = []
        class_lst = []
        savefig_path = os.path.join(config.output_path, 'casplot')
        savefig_path = os.path.abspath(savefig_path)
        if not os.path.exists(savefig_path):
            os.mkdir(savefig_path)

        if mAP:
            final_res = {}
            final_res['version'] = 'VERSION 1.3'
            final_res['results'] = {}
            final_res['external_data'] = {'used': True, 'details': 'Features from I3D Network'}

        num_correct = 0.
        num_total = 0.

        load_iter = iter(test_loader)

        for i in range(len(test_loader.dataset)):

            # GCN inner video merge version
            data, label, pseudo_label, vid_name, vid_num_seg = next(load_iter)
            data =data.reshape(1, 1, data.shape[-2], data.shape[-1]).cuda()
            pseudo_label = pseudo_label.reshape(1, 1, pseudo_label.shape[-2], pseudo_label.shape[-1]).cuda()
            label = label.reshape(-1, label.shape[-1]).cuda()

            gcn_data = torch.zeros_like(data)
            for index in torch.where(label[0]==1)[0]:
                cur_data, cur_nodes, cur_nodes_label = gcnn(data, pseudo_label, [index], eval=True)
                gcn_data += cur_data
                nodes_lst.append(torch.stack(cur_nodes)[0].detach().cpu().numpy())
                nodes_label_lst.append(torch.stack(cur_nodes_label)[0].detach().cpu().numpy())
                class_lst.append(np.ones(len(cur_nodes[0]))*(index.detach().cpu().item()))
            gcn_data = gcn_data/len(torch.where(label[0]==1)[0])

            data = torch.cat([data, gcn_data], dim=-1)
            # data = (data + gcn_data)/2
            data = data.reshape(-1, data.shape[-2], data.shape[-1]).detach()
            pseudo_label = pseudo_label.detach()

            # # normal version
            # data, label, _, vid_name, vid_num_seg = next(load_iter)
            # data = data.cuda()
            # label = label.cuda()

            vid_num_seg = vid_num_seg[0].cpu().item()
            num_segments = data.shape[1]
            score_act, _, feat_act, feat_bkg, features, cas_softmax, sup_cas_softmax = net(data)
            feat_magnitudes_act = torch.mean(torch.norm(feat_act, dim=2), dim=1)
            feat_magnitudes_bkg = torch.mean(torch.norm(feat_bkg, dim=2), dim=1)

            label_np = label.cpu().data.numpy()
            score_np = score_act[0].cpu().data.numpy()

            pred_np = np.zeros_like(score_np)
            pred_np[np.where(score_np < config.class_thresh)] = 0
            pred_np[np.where(score_np >= config.class_thresh)] = 1

            correct_pred = np.sum(label_np == pred_np, axis=1)

            num_correct += np.sum((correct_pred == config.num_classes).astype(np.float32))
            num_total += correct_pred.shape[0]

            feat_magnitudes = torch.norm(features, p=2, dim=2)

            feat_magnitudes = utils.minmax_norm(feat_magnitudes, max_val=feat_magnitudes_act, min_val=feat_magnitudes_bkg)
            feat_magnitudes = feat_magnitudes.repeat((config.num_classes, 1, 1)).permute(1, 2, 0)

            cas = utils.minmax_norm(cas_softmax * feat_magnitudes)

            # Plot and create feature dict
            # For WTAL head
            gt_vid = gt[vid_name[0]]
            cas = utils.get_cas(gt_vid, cas)
            wtal_pred_dict[vid_name[0]] = cas
            if save:
                plot_pred(cas, gt_vid, vid_name[0]+'_wtal_inner_', savefig_path)
            # For Supervision head
            if sup_cas_softmax is not None:
                cas = sup_cas_softmax
                cas = utils.get_cas(gt_vid, cas)
                sup_pred_dict[vid_name[0]] = cas
                if save:
                    plot_pred(cas, gt_vid, vid_name[0]+'_inner_seg_', savefig_path)

            if mAP:
                pred = np.where(score_np >= config.class_thresh)[0]

                if len(pred) == 0:
                    pred = np.array([np.argmax(score_np)])
                cas_pred = cas[0].cpu().numpy()[:, pred]
                cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))
                cas_pred = utils.upgrade_resolution(cas_pred, config.scale)

                proposal_dict = {}

                feat_magnitudes_np = feat_magnitudes[0].cpu().data.numpy()[:, pred]
                feat_magnitudes_np = np.reshape(feat_magnitudes_np, (num_segments, -1, 1))
                feat_magnitudes_np = utils.upgrade_resolution(feat_magnitudes_np, config.scale)

                for i in range(len(config.act_thresh_cas)):
                    cas_temp = cas_pred.copy()
                    zero_location = np.where(cas_temp[:, :, 0] < config.act_thresh_cas[i])
                    cas_temp[zero_location] = 0

                    seg_list = []
                    for c in range(len(pred)):
                        pos = np.where(cas_temp[:, c, 0] > 0)
                        seg_list.append(pos)

                    proposals = utils.get_proposal_oic(seg_list, cas_temp, score_np, pred, config.scale, \
                                    vid_num_seg, config.feature_fps, num_segments)

                    for i in range(len(proposals)):
                        class_id = proposals[i][0][0]

                        if class_id not in proposal_dict.keys():
                            proposal_dict[class_id] = []

                        proposal_dict[class_id] += proposals[i]

                for i in range(len(config.act_thresh_magnitudes)):
                    cas_temp = cas_pred.copy()

                    feat_magnitudes_np_temp = feat_magnitudes_np.copy()

                    zero_location = np.where(feat_magnitudes_np_temp[:, :, 0] < config.act_thresh_magnitudes[i])
                    feat_magnitudes_np_temp[zero_location] = 0

                    seg_list = []
                    for c in range(len(pred)):
                        pos = np.where(feat_magnitudes_np_temp[:, c, 0] > 0)
                        seg_list.append(pos)

                    proposals = utils.get_proposal_oic(seg_list, cas_temp, score_np, pred, config.scale, \
                                    vid_num_seg, config.feature_fps, num_segments)

                    for i in range(len(proposals)):
                        class_id = proposals[i][0][0]

                        if class_id not in proposal_dict.keys():
                            proposal_dict[class_id] = []

                        proposal_dict[class_id] += proposals[i]

                final_proposals = []
                for class_id in proposal_dict.keys():
                    final_proposals.append(utils.nms(proposal_dict[class_id], 0.6))

                final_res['results'][vid_name[0]] = utils.result2json(final_proposals)

        if save:
            plot_node(nodes_lst, nodes_label_lst, class_lst, os.path.abspath(config.output_path))

        if mAP:
            json_path = os.path.join(config.output_path, 'inner_result.json')
            with open(json_path, 'w') as f:
                json.dump(final_res, f)
                f.close()
            tIoU_thresh = np.linspace(0.1, 0.7, 7)
            anet_detection = ANETdetection(config.gt_path,
                                        json_path,
                                        subset='train',
                                        tiou_thresholds=tIoU_thresh,
                                        verbose=False,
                                        check_status=False)
            # mAP, average_mAP, fmAP, average_fmAP = anet_detection.evaluate()
            mAP, average_mAP = anet_detection.evaluate()

            test_acc = num_correct / num_total

        # mIoU
        if config.test_head == 'sup' and config.supervision != 'weak':
            print('IoU on sup head')
            test_iou, bkg_iou, act_iou = utils.calculate_iou(gt, sup_pred_dict, cls_thres)
        else:
            print('IoU on wtal head')
            test_iou, bkg_iou, act_iou = utils.calculate_iou(gt, wtal_pred_dict, cls_thres)

        # Update logger
        if mAP:
            logger.log_value('Test accuracy', test_acc, step)
            logger.log_value('Average mAP', average_mAP, step)
            for i in range(tIoU_thresh.shape[0]):
                logger.log_value('mAP@{:.1f}'.format(tIoU_thresh[i]), mAP[i], step)

        logger.log_value('Average mIoU', test_iou.mean(), step)
        for i in range(len(cls_thres)):
            logger.log_value('mIoU@{:.2f}_{:.2f}'.format(cls_thres[i][0], cls_thres[i][1]), test_iou[i], step)
            
        logger.log_value('Average bkg mIoU', bkg_iou.mean(), step)
        for i in range(len(cls_thres)):
            logger.log_value('bkg_mIoU@{:.2f}_{:.2f}'.format(cls_thres[i][0], cls_thres[i][1]), test_iou[i], step)

        logger.log_value('Average act mIoU', act_iou.mean(), step)
        for i in range(len(cls_thres)):
            logger.log_value('act_mIoU@{:.2f}_{:.2f}'.format(cls_thres[i][0], cls_thres[i][1]), test_iou[i], step)


        # Update test info
        test_info['step'].append(step)
        if mAP:
            test_info['test_acc'].append(test_acc)
            test_info['average_mAP'].append(average_mAP)
            for i in range(tIoU_thresh.shape[0]):
                test_info['mAP@{:.1f}'.format(tIoU_thresh[i])].append(mAP[i])

        test_info['average_mIoU'].append(test_iou.mean())
        for i in range(len(cls_thres)):
            test_info['mIoU@{:.2f}_{:.2f}'.format(cls_thres[i][0], cls_thres[i][1])].append(test_iou[i])

        test_info['average_bkg_mIoU'].append(bkg_iou.mean())
        for i in range(len(cls_thres)):
            test_info['bkg_mIoU@{:.2f}_{:.2f}'.format(cls_thres[i][0], cls_thres[i][1])].append(bkg_iou[i])

        test_info['average_act_mIoU'].append(act_iou.mean())
        for i in range(len(cls_thres)):
            test_info['act_mIoU@{:.2f}_{:.2f}'.format(cls_thres[i][0], cls_thres[i][1])].append(act_iou[i])

        if save:
            file_to_write = open(os.path.join(config.output_path,
                                              '{}_wtal_inner_pred_25.pickle'.format(config.test_dataset)), 'wb')
            pickle.dump(wtal_pred_dict, file_to_write)
            if sup_cas_softmax is not None:
                file_to_write = open(os.path.join(config.output_path,
                                              '{}_sup_inner_pred_25.pickle'.format(config.test_dataset)), 'wb')
                pickle.dump(sup_pred_dict, file_to_write)
        
    return sup_pred_dict
