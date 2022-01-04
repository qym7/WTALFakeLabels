'''
Author: your name
Date: 2021-12-22 20:30:23
LastEditTime: 2021-12-26 11:47:29
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /yimingqin/code/WTAL-Uncertainty-Modeling/options.py
'''
import argparse
import shutil
import os

def parse_args():
    descript = 'Pytorch Implementation of \'Weakly-supervsied Temporal Action Localization by Uncertainty Modeling\''
    parser = argparse.ArgumentParser(description=descript)

    parser.add_argument('--data_path', type=str, default='./dataset/THUMOS14')
    parser.add_argument('--model_path', type=str, default='./models/UM')
    parser.add_argument('--output_path', type=str, default='./outputs/UM')
    parser.add_argument('--log_path', type=str, default='./logs/UM')
    parser.add_argument('--modal', type=str, default='all', choices=['rgb', 'flow', 'all'])
    parser.add_argument('--alpha', type=float, default=0.0005)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--margin', type=int, default=100, help='maximum feature magnitude')
    parser.add_argument('--r_act', type=float, default=9)
    parser.add_argument('--r_bkg', type=float, default=4)
    parser.add_argument('--class_th', type=float, default=0.2)
    parser.add_argument('--lr', type=str, default='[0.0001]*10000', help='learning rates for steps(list form)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    parser.add_argument('--model_file', type=str, default=None, help='the path of pre-trained model file')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test_dataset', default='test')
    parser.add_argument('--supervision', default='weak')
    parser.add_argument('--test_head', default='sup')
    parser.add_argument('--supervision_path', default='')
    parser.add_argument('--thres', default=0.2, type=float)
    parser.add_argument('--lmbd', default=0.1, type=float)
    parser.add_argument('--bkg_lmbd', default=1, type=float)
    parser.add_argument('--save', default=0, type=int)
    parser.add_argument('--m', default=0.9, type=float, help='decay params for EMA')
    parser.add_argument('--gamma', default=1.0, type=float, help='loss weight for EMA')

    return init_args(parser.parse_args())


def init_args(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if os.path.exists(args.log_path):
        shutil.rmtree(args.log_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args
