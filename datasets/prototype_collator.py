"""
原型网络 Collator。

将 sampler 按 [support..., query...] 分组的索引整理为模型所需的 support / query 特征与标签。
"""

from collections import defaultdict

import torch
import random


class PrototypeCollator:

    def __init__(self, protein_feats, sampler, n_way, n_query, n_support, num_classes, support):
        """
        Args:
            protein_feats: dict[int -> Tensor]，蛋白质索引到特征的映射
            sampler:       PrototypeSampler，用于读取 last_go_terms
            n_way:   每 batch 的 GO term 数
            n_query: 每个 GO term 的查询蛋白质数
            n_support: 每个 GO term 的支持蛋白质数
            num_classes: 总 GO term 数
        """
        self.protein_feats = protein_feats
        self.sampler = sampler
        self.n_way = n_way
        self.n_query = n_query
        self.n_support = n_support
        self.num_classes = num_classes
        self.group_size = n_support + n_query  # 每个 GO term 的总蛋白质数
        self.support = support

    def __call__(self, data):
        """
        data: list of (idx, seq, label_indices)，按 sampler 的顺序排列：
              每 group_size 个元素属于同一个 GO term，前 n_support 个是 support，后 n_query 个是 query。
        """
        inputs_feats = torch.stack([self.protein_feats[line[0]] for line in data])
        labels = self._get_labels([line[2] for line in data])
        
        support_indices = []
        for c in range(len(self.support)):
            pool = self.support[c]
            if len(pool) < self.n_support:
                indices =  random.sample(pool, len(pool)) + random.choices(pool, k=self.n_support - len(pool))
            else:
                indices =  random.sample(pool, self.n_support)
            support_indices.append(indices)
        
        data2classes = defaultdict(set) # key是数据行数，value是类别索引集合，用于计算原型时相互映射
        for outer_idx, inner_list in enumerate(support_indices):
            for element in inner_list:
                data2classes[element].add(outer_idx)

        unique_proteins = list(data2classes.keys())
        support_feats = torch.stack([self.protein_feats[d] for d in unique_proteins])  # 用于计算原型的所有特征

        protein_to_idx = {protein: i for i, protein in enumerate(unique_proteins)}

        return inputs_feats, labels, support_feats, data2classes, torch.stack([torch.tensor([protein_to_idx[c] for c in c_list]) for c_list in support_indices], dim=0) # (每个类的支持蛋白质索引映射到 support_feats 的索引)

    def _get_labels(self, label_indices_list):
        """将每个蛋白质的 GO term 索引列表转换为二值标签向量。"""
        labels = torch.zeros(len(label_indices_list), self.num_classes, dtype=torch.float32)
        for i, label_indices in enumerate(label_indices_list):
            labels[i, label_indices] = 1.0
        return labels
