"""
原型网络类别平衡采样器。

每 batch 随机采样 n_way 个 GO term，对每个 term 采样 (n_query + n_support) 个正样本蛋白质。
返回的索引列表按 GO term 分组：每组先 n_support 个（用于计算原型），后 n_query 个（查询样本）。
"""

import random
import torch


class PrototypeSampler:
    """
    采样格式：每 batch 输出 [c0_s0, ..., c0_s_{M-1}, c0_q0, ..., c0_q_{N-1},
                          c1_s0, ..., c1_s_{M-1}, c1_q0, ..., c1_q_{N-1}, ...]
    即每个 GO term：先 M 个 support，后 N 个 query。
    """

    def __init__(self, query, n_way, n_query):
        """
        Args:
            query: Tensor[num_classes, num_proteins]，二值矩阵，query[c][p]=1 表示蛋白质 p 有 GO term c
            n_way:  每 batch 采样的 GO term 数量
            n_query:   每个 GO term 的查询蛋白质数量
            n_support: 每个 GO term 的支持蛋白质数量（用于计算原型）
        """
        self.query = query
        self.n_way = n_way
        self.n_query = n_query
        self.valid_classes = [i for i, sublist in enumerate(query) if sublist]

    def _sample_for_class(self, c):
        """为类别 c 采样查询集,不足的有放回采样。"""
        pool = self.query[c]
        if len(pool) >= self.n_query:
            return random.sample(pool, self.n_query)
        else:
            return random.sample(pool, len(pool)) + random.choices(pool, k=self.n_query - len(pool))

    def __iter__(self):
        for _ in range(len(self)):
            choosed_classes_indices = random.sample(self.valid_classes, self.n_way)
            batch = []
            for c in choosed_classes_indices:
                batch.extend(self._sample_for_class(c))
            yield batch

    def __len__(self):
        return len(self.valid_classes) // self.n_way
