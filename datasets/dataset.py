import os
import pickle
import random
from unittest import result
import torch
from torch.utils.data import Dataset
import numpy as np

class Dataset(Dataset):
    def __init__(self, dataset_path, namespace, train_mode, dataset_mode, val_ratio=0.1, valid_mask=None):

        assert 0 <= val_ratio < 1, "val_ratio must be between 0 and 1"

        seq_path = os.path.join(dataset_path, f"{dataset_mode}_seq_{namespace.lower()}")
        label_path = os.path.join(dataset_path, f"{dataset_mode}_label_{namespace.lower()}")

        with open(seq_path, 'rb') as f:
            seqs = pickle.load(f)
        with open(label_path, 'rb') as f:
            labels = pickle.load(f)

        assert len(seqs) == len(labels), \
            f"seq ({len(seqs)}) and label ({len(labels)}) count mismatch for {dataset_mode}/{namespace}"

        # self.data: list of (idx, sequence, label_indices)
        # data = [
        #     (i, seqs[i]['seq'], labels[i])
        #     for i in range(len(seqs))
        # ]
        data = [
            (i, seqs[i]['seq'], labels[i]) if len(labels[i])!=0 else (i, seqs[i]['seq'], [0]) # # 蛋白质数据集中有几条数据未标注namespace根节点功能(CC中是3条)
            for i in range(len(seqs)) 
        ]
        if valid_mask is not None: data = self.update_labels_numpy(valid_mask, data) # 去除无效标签，并更新标签索引

        # 从训练集中随机划分验证集
        if train_mode == 'train':
            random.seed(42)
            random.shuffle(data)
            self.split_idx = int(len(data) * (1 - val_ratio))
            self.data = data[:self.split_idx]
            self._val_data = data[self.split_idx:]
        else:
            self.data = data
            self._val_data = None
        
    @property
    def val_dataset(self):
        if self._val_data is None:
            return None
        ds = Dataset.__new__(Dataset)
        ds.data = self._val_data
        ds._val_data = None
        return ds

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _map_indices(self, tuple_list):
        # 1. 找出所有B中最大的元素，以此确定输出列表的长度（最大值 + 1）
        max_val = max((x for tup in tuple_list for x in tup[2]), default=-1)
        if max_val < 0:
            return []
        
        # 2. 初始化目标列表，长度为 max_val + 1
        result = [[] for _ in range(max_val + 1)]
        
        # 3. 遍历A，将A的索引填入对应位置
        for idx, tup in enumerate(tuple_list):
            for val in tup[2]:
                result[val].append(idx)
                
        return result
    
    def update_labels_numpy(self, valid_mask, data):
        # 将 valid_mask 转换为布尔类型的 numpy 数组
        mask = np.array(valid_mask, dtype=bool)
        
        # 1. 计算前缀和：计算到当前位置为止，有多少个 True
        # 例如：[True, True, False, True] -> [1, 2, 2, 3]
        # 减去 1 转换为 0 开始的索引 -> [0, 1, 1, 2]
        mapping = np.cumsum(mask) - 1
        
        new_data = []
        for row_idx, data_str, label_list in data:
            new_label_list = []
            for label in label_list:
                # 确保标签在 valid_mask 范围内，且该标签本身没有被 mask 掉
                if label < len(mask) and mask[label]:
                    # 通过映射表直接获取新索引
                    new_label_list.append(int(mapping[label]))
            new_data.append((row_idx, data_str, new_label_list))
            
        return new_data