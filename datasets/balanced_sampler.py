import random
import torch
from torch.utils.data import BatchSampler
import torch.nn.functional as F

class ClassBalanceSampler(BatchSampler):
    """
    每 batch 随机选 N 个类别，每类采 k 个正样本 + k 个负样本。
    返回 batch_size = 2 * N * k 的索引列表。
    """

    def __init__(self, pos_indices, easy_neg_indices, hard_neg_indices, num_classes, N=32, k=4,
                 drop_last=False):
        """
        Args:
            pos_indices: Tensor[C, *] 或 list[list[int]] — 每类的正样本全局索引
            neg_indices: Tensor[C, *] 或 list[list[int]] — 每类的负样本全局索引
            num_classes: int
            N: 每 batch 采样的类别数
            k: 每类正/负样本数
        """
        self.pos_indices = pos_indices
        self.easy_neg_indices = easy_neg_indices
        self.hard_neg_indices = hard_neg_indices
        self.num_classes = num_classes
        self.N = N
        self.k = k
        self.drop_last = drop_last
        self.valid_classes = torch.nonzero(self.pos_indices.sum(1)).squeeze(1)

    def __iter__(self):
        for _ in range(len(self)):
            batch_indices = []
            # 随机选 N 个有效类别
            chosen_classes = random.sample(self.valid_classes.tolist(), self.N)

            for c in chosen_classes:
                # 正样本
                pos_pool = torch.nonzero(self.pos_indices[c]).squeeze(1)
                pos_idx = torch.randperm(len(pos_pool))
                if len(pos_pool) >= self.k:
                    batch_indices.extend(pos_pool[pos_idx[:self.k]].tolist())
                else:
                    # 真实的索引
                    batch_indices.extend(pos_pool[pos_idx].tolist())
                    # 填充 -1
                    batch_indices.extend([-1] * (self.k - len(pos_idx)))

                # 负样本
                easy_neg_pool = torch.nonzero(self.easy_neg_indices[c]).squeeze(1)
                easy_neg_idx = torch.randperm(len(easy_neg_pool))

                hard_neg_pool = torch.nonzero(self.hard_neg_indices[c]).squeeze(1)
                hard_neg_idx = torch.randperm(len(hard_neg_pool))
                if len(hard_neg_pool) >= self.k//2:
                    if len(easy_neg_pool) >= self.k//2:
                        batch_indices.extend(hard_neg_pool[hard_neg_idx[:self.k//2]].tolist() + easy_neg_pool[easy_neg_idx[:self.k//2]].tolist())
                    else:
                        index_list = hard_neg_pool[hard_neg_idx[:(self.k-len(easy_neg_idx))]].tolist() + easy_neg_pool[easy_neg_idx].tolist()
                        batch_indices.extend(index_list)
                        batch_indices.extend([-1] * (self.k - len(index_list)))
                else:
                    index_list = hard_neg_pool[hard_neg_idx].tolist() + easy_neg_pool[easy_neg_idx[:(self.k-len(hard_neg_idx))]].tolist()
                    batch_indices.extend(index_list)
                    batch_indices.extend([-1] * (self.k - len(index_list)))

            yield batch_indices

    def __len__(self):
        # 每个 epoch 的 batch 数
        return self.num_classes // self.N