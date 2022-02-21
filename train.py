import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

import utils
from config import NODES_NUMBER


def train(net, gcnn, loader_iter, optimizer, optimizer_gcnn,
          criterion, criterion_gcnn, logger, step, gcnn_teacher,
          m, nodes_bank):
    net.train()

    data, label, gt, vid_names, _, index = next(loader_iter)
    data = data.cuda()  # reshaped in net
    gt = gt.cuda()  # reshaped in net
    label = label.cuda()
    index = index.cuda()

    # Get GCNN feature
    gcn_data, nodes, nodes_label = gcnn(data, gt, index, eval=False)
    # Update nodes_bank
    # Do not use grad to accelerate algorithm
    
    with torch.no_grad():
        _, t_nodes, t_nodes_label = gcnn_teacher(data, gt, index, eval=False)
        if step > 20:
            # filter a cleaner node bank when node bank is not empty
            _, t_nodes, t_nodes_label = criterion_gcnn(t_nodes, t_nodes_label, index, nodes_bank)
        
    for i in range(len(index)):
        not_torch_idx = index[i].detach().cpu().item()
        nodes_number = NODES_NUMBER[not_torch_idx]
        # filter4
        k = 25
        random_act_idx = torch.randperm(len(t_nodes))[:k]
        random_bkg_idx = torch.randperm(len(t_nodes))[:k]
        t_act_nodes = t_nodes[i][t_nodes_label[i]==2][random_act_idx]
        t_bkg_nodes = t_nodes[i][t_nodes_label[i]==0][random_bkg_idx]
        # update action nodes
        if len(nodes_bank[not_torch_idx]) == 0:
            nodes_bank[not_torch_idx] = t_act_nodes
        else:
            nodes_bank[not_torch_idx] = torch.cat((t_act_nodes, nodes_bank[not_torch_idx]))
        nodes_bank[not_torch_idx] = nodes_bank[not_torch_idx][:nodes_number]
        # update bkg nodes
        if len(nodes_bank[20]) == 0:
            nodes_bank[20] = t_bkg_nodes
        else:
            nodes_bank[20] = torch.cat((t_bkg_nodes, nodes_bank[20]))
        nodes_bank[20] = nodes_bank[20][:NODES_NUMBER[20]]

    # Calculate Contrastive Loss and Back-propagate GCNN
    cost_gcnn, _, _ = criterion_gcnn(nodes, nodes_label, index, nodes_bank)
    optimizer_gcnn.zero_grad()
    cost_gcnn.backward()
    optimizer_gcnn.step()

    # Isolate gradient between GCNN and WTAL model
    label = label.reshape(-1, label.shape[-1]).to(torch.float32).detach()
    data = torch.cat([data, gcn_data], dim=-1)
    data = data.reshape(-1, data.shape[-2], data.shape[-1]).detach()
    gt = gt.reshape(-1, gt.shape[-2], gt.shape[-1]).detach()
    score_act, score_bkg, feat_act, feat_bkg, _, _, sup_cas_softmax = net(data)

    # Calculate WTAL Loss and Back-propagate
    cost, loss = criterion(score_act, score_bkg, feat_act, feat_bkg, label,
                           gt, sup_cas_softmax, step=step)
    loss['loss_gcnn'] = cost_gcnn
    print("loss_gcnn", cost_gcnn.detach().cpu().item())

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if gcnn_teacher is not None:
        # update teacher parameters by EMA
        student_params = OrderedDict(gcnn.named_parameters())
        teacher_params = OrderedDict(gcnn_teacher.named_parameters())

        # check if both model contains the same set of keys
        assert student_params.keys() == teacher_params.keys()

        for name, param in student_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            teacher_params[name] = teacher_params[name] * m + (1 - m) * param

        student_buffers = OrderedDict(gcnn.named_buffers())
        teacher_buffers = OrderedDict(gcnn_teacher.named_buffers())

        # check if both model contains the same set of keys
        assert student_buffers.keys() == teacher_buffers.keys()

        for name, buffer in student_buffers.items():
            teacher_buffers[name] = teacher_buffers[name] * m + (1 - m) * buffer

    for key in loss.keys():
        logger.log_value(key, loss[key].cpu().item(), step)
        

