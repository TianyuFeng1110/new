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

    @property
    def prototype_feats(self):
        """根据训练集(train mode为根据val_ration划分的新训练集，test mode则为完整的训练集)具体包括的go term，从预计算的蛋白质特征中提取原型特征。
        TODO 原型的逻辑还需要再思考一下，目前用ESM2的池化特征作为原型特征的输入，但训练集中有的go term可能没有对应的蛋白质，这时就无法计算原型特征了(这块是必须解决的问题)。按照逻辑原型可以用自己训练好的特征，而不仅仅是ESM2的特征。
        """
        return None
        # protein_idx = self._map_indices(self.data) # 用于计算原型每个元素表示go term对应的蛋白质列表(存的是索引)
        # prototype_feats = []
        # for idxs in protein_idx:
        #     proteins_list = []
        #     for idx in idxs:
        #         proteins_list.append(self.protein_feats[idx])
        #     if len(proteins_list) > 0:
        #         prototype_feats.append(torch.stack(proteins_list).mean(dim=0)) # 计算均值作为原型特征
        #     else:
        #         prototype_feats.append(None) # 训练集中如果某个go term没有对应的蛋白质，也就无法计算原型，设为None
        # return prototype_feats

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