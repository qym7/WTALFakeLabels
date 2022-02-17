import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

import utils


def train(net, gcnn, loader_iter, optimizer, optimizer_gcnn, criterion, criterion_gcnn, logger, step, net_teacher, m, nodes_dict):
    net.train()

    data, nodes, nodes_label, nodes_pos, nodes_nbr, label, pseudo_label, vid_names, _, index = next(loader_iter)
    data = data.cuda()
    nodes = nodes.cuda()
    nodes_label = nodes_label.cuda()
    nodes_pos = nodes_pos.cuda()
    nodes_nbr = nodes_nbr.cuda()
    label = label.cuda()
    pseudo_label = pseudo_label.cuda()

    # Calculate GCNN contrastive loss
    # nodes, gcn_data = gcnn(nodes, nodes_label, nodes_pos, nodes_nbr, data)
    nodes, gcn_data = gcnn(data, pseudo_labels, label)
    cost_gcnn = criterion_gcnn(nodes, nodes_label, nodes_nbr)

    optimizer_gcnn.zero_grad()
    cost_gcnn.backward()
    optimizer_gcnn.step()

    # Isolate GCNN gradient and NET gradient
    label = label.reshape(-1, label.shape[-1]).detach()
    pseudo_label = pseudo_label.reshape(label.shape[0], -1,
                                        pseudo_label.shape[-1]).detach()
    # data = torch.cat([gcn_data, data], dim=-1)
    # data = data.reshape(label.shape[0], -1, data.shape[-1]).detach()
    data = data.reshape(label.shape[0], -1, data.shape[-1]).detach()
    # print(data.shape)
    # data = gcn_data.reshape(label.shape[0], -1, gcn_data.shape[-1]).detach()

    # Calculate WTAL loss
    score_act, score_bkg, feat_act, feat_bkg, _, _, sup_cas_softmax = net(data)

    cost, loss = criterion(score_act, score_bkg, feat_act, feat_bkg, label,
                           pseudo_label, sup_cas_softmax, step=step)
    loss['loss_gcnn'] = cost_gcnn
    print("loss_gcnn", cost_gcnn.detach().cpu().item())

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

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
        

