import torch.utils.data as data
import os
import csv
import json
import pickle
import numpy as np
import torch
import pdb
import time
import random
import utils
import config


class GCNThumosFeature(data.Dataset):
    def __init__(self, data_path, mode, modal, feature_fps, num_segments, sampling, seed=-1,
                 supervision='weak', supervision_path=None, N=1):
        if seed >= 0:
            utils.set_seed(seed)

        self.mode = mode if mode == 'test' else 'train'
        self.modal = modal
        self.feature_fps = feature_fps
        self.num_segments = num_segments
        _mode = 'test' if self.mode == 'test' else 'val'

        if self.modal == 'all':
            self.feature_path = []
            for _modal in ['rgb', 'flow']:
                self.feature_path.append(os.path.join(data_path, 'features', _mode, _modal))
        else:
            self.feature_path = os.path.join(data_path, 'features', _mode, self.modal)

        split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.strip())
        split_file.close()

        anno_path = os.path.join(data_path, 'gt.json')
        anno_file = open(anno_path, 'r')
        self.anno = json.load(anno_file)
        anno_file.close()
        
        self.supervision = supervision
        if self.supervision != 'weak':
            if len(supervision_path) == 0:
                raise('Supervision path is not given.')
            anno_file = open(supervision_path, 'rb')
            self.temp_annots = pickle.load(anno_file)
            anno_file.close()

        self.class_name_to_idx = dict((v, k) for k, v in config.class_dict.items())        
        self.num_classes = len(self.class_name_to_idx.keys())
        self.all_labels, self.label_group = self.get_all_labels()

        self.sampling = sampling
        self.N = N

    def __len__(self):
        return len(self.label_group)

    def __getitem__(self, index):
        v_index = np.random.choice(self.label_group[index], self.N)  # 随机选取index类别中的self.N个视频
        data = []
        label = []
        temp_anno = []
        vid_name = []
        vid_num_seg = []
        for i, idx in enumerate(v_index):
            data_, label_, temp_anno_, vid_name_, vid_num_seg_ = self.get_single_item(idx)
            data.append(data_)
            label_ = np.zeros([self.num_classes], dtype=np.float32)
            label_[index] = 1
            label.append(label_)
            temp_anno.append(temp_anno_)
            vid_name.append(vid_name_)
            vid_num_seg.append(vid_num_seg_)

        return np.stack(data), np.stack(label), np.stack(temp_anno), vid_name, vid_num_seg, index

    def get_single_item(self, index):
        data, vid_num_seg, sample_idx = self.get_data(index)
        label, temp_anno = self.get_label(index, vid_num_seg, sample_idx)
        return data, label, temp_anno, self.vid_list[index], vid_num_seg

    def get_all_labels(self):
        all_labels = []
        label_group = {k: [] for k in config.class_dict}
        for i in range(len(self.vid_list)):
            vid_name = self.vid_list[i]
            label = np.zeros([self.num_classes], dtype=np.float32)
            anno_list = self.anno['database'][vid_name]['annotations']
            for _anno in anno_list:
                label_idx = self.class_name_to_idx[_anno['label']]
                label[label_idx] = 1
                if i not in label_group[label_idx]:
                    label_group[label_idx].append(i)
            all_labels.append(label)

        return all_labels, label_group

    def get_data(self, index):
        vid_name = self.vid_list[index]

        vid_num_seg = 0

        if self.modal == 'all':
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                    vid_name + '.npy')).astype(np.float32)
            flow_feature = np.load(os.path.join(self.feature_path[1],
                                    vid_name + '.npy')).astype(np.float32)
            vid_num_seg = rgb_feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(rgb_feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(rgb_feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            rgb_feature = rgb_feature[sample_idx]
            flow_feature = flow_feature[sample_idx]

            feature = np.concatenate((rgb_feature, flow_feature), axis=1)
        else:
            feature = np.load(os.path.join(self.feature_path,
                                    vid_name + '.npy')).astype(np.float32)

            vid_num_seg = feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            feature = feature[sample_idx]

        return feature, vid_num_seg, sample_idx

    def get_label(self, index, vid_num_seg, sample_idx):
        vid_name = self.vid_list[index]
        anno_list = self.anno['database'][vid_name]['annotations']
        label = self.all_labels[index]

        classwise_anno = [[]] * self.num_classes

        for _anno in anno_list:
            classwise_anno[self.class_name_to_idx[_anno['label']]].append(_anno)

        if self.supervision == 'weak':
            return label, torch.Tensor(0)
        else:
            temp_annot = self.temp_annots[vid_name]
            temp_annot = temp_annot[sample_idx]

            return label, temp_annot

    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        ######################## ema code to be deleted
        samples = np.arange(self.num_segments) * length / self.num_segments
        return np.floor(samples).astype(int)
        ########################
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        if length <= self.num_segments:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)

    def regenerate(self, new_pseudo_labels, update_method='ema'):
        '''
        Update current pseudolabels through new_pseudo_labels
        '''
        for vid_name in self.temp_annots:
            if update_method == 'ema':
                self.temp_annots[vid_name] =  new_pseudo_labels[vid_name] * 0.01 + \
                    self.temp_annots[vid_name] + 0.99
            else:
                self.temp_annots[vid_name] = new_pseudo_labels
                
        
    
