import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

import utils


def train(net, gcnn, loader_iter, optimizer, optimizer_gcnn, criterion, criterion_gcnn, logger, step, net_teacher, m, nodes_dict):
    net.train()

    data, label, gt, vid_names, _, index = next(loader_iter)
    data = data.cuda()  # reshaped in net
    gt = gt.cuda()  # reshaped in net
    label = label.reshape(-1, label.shape[-1]).cuda()

    data, nodes, nodes_label, vids_label = gcnn(data, gt, index, vid_names)
    cost_gcnn = criterion_gcnn(nodes, nodes_label)

    data = data.detach()
    gt = gt.reshape(-1, gt.shape[-2], gt.shape[-1]).detach()
    score_act, score_bkg, feat_act, feat_bkg, _, _, sup_cas_softmax = net(data)

    cost, loss = criterion(score_act, score_bkg, feat_act, feat_bkg, label,
                           gt, sup_cas_softmax, step=step)
    loss['loss_gcnn'] = cost_gcnn
    print("loss_gcnn", cost_gcnn.detach().cpu().item())

    optimizer.zero_grad()
    optimizer_gcnn.zero_grad()
    # cost_gcnn.backward()
    cost = cost + cost_gcnn
    cost.backward()
    optimizer.step()
    optimizer_gcnn.step()

    if net_teacher is not None:
        # update teacher parameters by EMA
        student_params = OrderedDict(net.named_parameters())
        teacher_params = OrderedDict(net_teacher.named_parameters())

        # check if both model contains the same set of keys
        assert student_params.keys() == teacher_params.keys()

        for name, param in student_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            teacher_params[name] = teacher_params[name] * m + (1 - m) * param

        student_buffers = OrderedDict(net.named_buffers())
        teacher_buffers = OrderedDict(net_teacher.named_buffers())

        # check if both model contains the same set of keys
        assert student_buffers.keys() == teacher_buffers.keys()

        for name, buffer in student_buffers.items():
            teacher_buffers[name] = teacher_buffers[name] * m + (1 - m) * buffer

    for key in loss.keys():
        logger.log_value(key, loss[key].cpu().item(), step)
        

