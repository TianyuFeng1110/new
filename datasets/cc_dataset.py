import os
import pickle
import random
from unittest import result
import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, dataset_path, namespace, train_mode, dataset_mode, val_ratio=0.1):

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
        data = [
            (i, seqs[i]['seq'], labels[i])
            for i in range(len(seqs))
        ]

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