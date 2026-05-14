

import torch
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import random
from Bio import SeqIO

class collator:
    def __init__(self, go2id, protein_feats):
        self.go2id = go2id
        self.protein_feats = protein_feats
        
    def __call__(self, data):
        inputs = torch.stack([self.protein_feats[elem[0]] for elem in data])
        labels = self._get_labels(data)

        return inputs, labels

    def _get_labels(self, data):
        num_classes = len(self.go2id)

        # 初始化全 0 tensor
        multi_hot = torch.zeros((len(data), num_classes))

        for i, (_, index_set) in enumerate(data):
            if index_set:  # 确保集合不为空
                # 将 set 转为 list，一次性将该行对应的多个位置设为 1
                multi_hot[i, [self.go2id[go] for go in index_set if go in self.go2id]] = 1.0
                
        return multi_hot