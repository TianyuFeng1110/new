"""
类平衡采样 Collator + 按类平均损失。

采样策略：
  每 batch 随机选 N 个 GO term 类别 → 每类采 k 个正样本 + k 个负样本
  → 不足 k 时全取并用全零向量补齐 → 生成 valid_mask 标记有效位
  → 损失按「类别」平均，使每个类对梯度的贡献 1:1。
"""

import random
import torch


class BalancedCollator:

    def __init__(self, data, protein_feats, num_classes):
        """
        Args:
            pos_indices:  list[Tensor]  pos_indices[c] = 类 c 正样本的全局索引
            neg_indices:  list[Tensor]  neg_indices[c] = 类 c 负样本的全局索引
            label_sets:   list[set]     label_sets[idx] = 样本 idx 的标签类别集合
            protein_feats: dict[int -> Tensor]  蛋白质特征字典
            num_classes:  int           总类别数
            N:            int           每 batch 采样的类数
            k:            int           每类正 / 负样本数
            sampler:      ClassBalanceSampler  用于读取 last_batch_indices（num_workers=0 时安全）
        """
        self.data = data
        self.protein_feats = protein_feats
        self.num_classes = num_classes

    def __call__(self, data):
        inputs = torch.stack([
            self.protein_feats[p[0]] if p[0] != -1 else torch.zeros_like(self.protein_feats[0])
            for p in data
        ])

        labels = self._get_labels(data)
        mask = torch.tensor([triplet[0] != -1 for triplet in data]) # 计算损失时要去掉mask为False的位置。

        return inputs, labels, mask

    def _get_labels(self, data):
        """将整数标签索引列表转为 multi-hot 向量。

        Args:
            data: list of (idx, mode, seq, label_indices, go_ids)

        Returns:
            multi_hot: tensor[B, num_classes]，值为 0.0 或 1.0
        """
        multi_hot = torch.zeros((len(data), self.num_classes))

        for i, (_, _, label_indices) in enumerate(data):
            if label_indices:
                multi_hot[i, label_indices] = 1.0

        return multi_hot