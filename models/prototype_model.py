"""
原型网络模型。

- 随机初始化的 MLP 将蛋白质特征映射到嵌入空间
- Support 蛋白质的嵌入按 GO term 平均得到原型
- Query 蛋白质嵌入与各原型的负欧氏距离作为预测 logits
- 使用 BCE 损失反向更新 MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import AsymmetricLossMultiLabel


class PrototypeNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, frequency,  parents_matrix, class_counts, proto_w, dropout=0.4, smooth_tau=30):
        super().__init__()

        self.register_buffer('parents_matrix', torch.as_tensor(parents_matrix, dtype=torch.float32))
        self.register_buffer('frequency', frequency.float())
        self.register_buffer('class_counts', class_counts.float())
        self.register_buffer('smooth_tau', torch.tensor(smooth_tau, dtype=torch.float32))
        self.register_buffer('proto_w', torch.as_tensor(proto_w, dtype=torch.float32))

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.temperature = nn.Parameter(torch.ones(num_classes) * 2.0)
        # 用 logit(先验概率) 初始化 bias，而非概率本身
        eps = 1e-6
        freq_clamped = frequency.clamp(min=eps, max=1 - eps)
        self.bias = nn.Parameter(torch.log(freq_clamped / (1 - freq_clamped)))

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4, gamma_pos=0, clip=0.0, eps=1e-8,
        )

    def forward(self, query, labels, support=None, support_indices=None, prototypes=None): # query的数量是固定的，不足的是有放回的
        
        q_feats = self.mlp(query)         # (num_queries, hidden_dim)

        if self.training:
            s_feats = self.mlp(support)       # (num_support, hidden_dim)
            gathered = s_feats[support_indices]        # (num_class, support_num, hidden_dim)
            proto_feats = gathered.mean(dim=1)           # (num_class, hidden_dim)
            proto_feats = self._smooth_prototypes(proto_feats)
            logits = self.predict(q_feats, proto_feats)  # (num_queries, num_class)
        else:
             logits = self.predict(q_feats, prototypes) 
        loss = self.loss_fn(logits, labels)

        return logits, loss, self.temperature.mean()
    
    def _smooth_prototypes(self, self_proto):
        """频率加权平滑：w * 自身原型 + (1 - w) * 祖先原型加权平均。

        祖先原型 = 按 proto_w 权重对所有祖先原型做加权平均（归一化后）。
        罕见功能（正样本少）更多依赖稳定祖先原型；根节点（无祖先）回退为自身原型。

        Args:
            self_proto: (num_classes, hidden_dim) 纯均值原型
        Returns:
            (num_classes, hidden_dim) 平滑后的原型
        """
        if self.parents_matrix is None:
            return self_proto

        # proto_w: (num_classes, num_classes)，proto_w[i][j] 是节点 i 对祖先 j 的权重
        # 按权重对所有祖先原型加权平均
        weight_sum = self.proto_w.sum(dim=1, keepdim=True)                          # (num_classes, 1)
        has_ancestor = weight_sum.squeeze(1) > 0                                    # (num_classes,)
        stable_ancestor_proto = self.proto_w @ self_proto                           # (num_classes, hidden_dim) 加权和
        stable_ancestor_proto = stable_ancestor_proto / weight_sum.clamp(min=1e-8)  # 归一化为加权平均

        # 无祖先的根节点：回退为自身原型
        stable_ancestor_proto[~has_ancestor] = self_proto[~has_ancestor]

        w = (self.class_counts / (self.class_counts + self.smooth_tau)).unsqueeze(1) # (num_classes, 1) zero-shot的功能由于class_counts为0，计算出的w为0，因此原型刚好是由其祖先加权得来的
        return w * self_proto + (1.0 - w) * stable_ancestor_proto

    def predict(self, query_emb, prototypes):
        q = F.normalize(query_emb, p=2, dim=1)      # 单位球面上
        p = F.normalize(prototypes, p=2, dim=1)
        cos_sim = q @ p.T                           # 范围 [-1, 1]
        temp = F.softplus(self.temperature) + 1e-5 
        return temp * cos_sim + self.bias          # 可学习的 temperature
    
    @ torch.no_grad()
    def _get_prototypes(self, all_feats, prototype_index):
        self.eval()

        s_feats = self.mlp(all_feats)         # (B, D)
        proto_mask = prototype_index.float()               # (B, C)
        proto_feats = proto_mask.T @ s_feats               # (C, D): 每个类累加其所有正样本蛋白质的嵌入
        class_counts = proto_mask.sum(dim=0).clamp(min=1)  # (C,):  每个类的正样本数
        proto_feats = proto_feats / class_counts.unsqueeze(1)  # (C, D): 均值 → 原型
        proto_feats = self._smooth_prototypes(proto_feats)
        return proto_feats