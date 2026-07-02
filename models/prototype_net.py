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


class PrototypeNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0.4):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, support_feats, support_sizes, query_feats):
        """
        Args:
            support_feats: (n_way, n_support, feat_dim) — 原始蛋白质特征
            support_sizes: (n_way,) — 每个 GO term 的实际 support 数量
            query_feats:   (num_queries, feat_dim)
        Returns:
            logits:  (num_queries, n_way) — 每个 query 对每个原型的预测 logits
            loss:    scalar — BCE loss（若提供 query_labels）
        """
        n_way, n_support, feat_dim = support_feats.shape
        num_queries = query_feats.shape[0]

        # 1. 所有蛋白质通过 MLP 映射
        support_flat = support_feats.view(n_way * n_support, feat_dim)
        all_feats = torch.cat([support_flat, query_feats], dim=0)
        all_emb = self.mlp(all_feats)

        support_emb = all_emb[:n_way * n_support].view(n_way, n_support, -1)
        query_emb = all_emb[n_way * n_support:]

        # 2. 平均 support 嵌入 → 原型: (n_way, hidden_dim)
        mask = torch.arange(n_support, device=support_feats.device).unsqueeze(0) < support_sizes.unsqueeze(1)
        support_emb = support_emb * mask.unsqueeze(-1)
        prototypes = support_emb.sum(dim=1) / support_sizes.clamp(min=1).unsqueeze(-1)

        # 3. 负平方欧氏距离 → logits: (num_queries, n_way)
        # ||q - p||^2 = ||q||^2 + ||p||^2 - 2*q·p
        q_norm = query_emb.pow(2).sum(1, keepdim=True)   # (num_queries, 1)
        p_norm = prototypes.pow(2).sum(1)                 # (n_way,)
        dist_sq = q_norm + p_norm.unsqueeze(0) - 2 * (query_emb @ prototypes.T)
        logits = -dist_sq

        return logits

    def encode(self, feats):
        """将原始特征通过 MLP 映射到嵌入空间。"""
        return self.mlp(feats)

    def predict(self, query_emb, prototypes):
        """给定 query 嵌入和原型，返回负欧氏距离 logits。
        Args:
            query_emb:  (num_queries, hidden_dim)
            prototypes: (num_classes, hidden_dim)
        Returns:
            logits: (num_queries, num_classes)
        """
        q_norm = query_emb.pow(2).sum(1, keepdim=True)
        p_norm = prototypes.pow(2).sum(1)
        dist_sq = q_norm + p_norm.unsqueeze(0) - 2 * (query_emb @ prototypes.T)
        return -dist_sq

    def compute_loss(self, logits, query_labels):
        """BCE loss。"""
        mask = query_labels.sum(1) > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return F.binary_cross_entropy_with_logits(
            logits[mask], query_labels[mask]
        )


@torch.no_grad()
def compute_all_prototypes(model, protein_feats, pos_indices, device):
    """用全部训练蛋白质计算所有 GO term 的原型。
    Args:
        model:         PrototypeNet（eval 模式）
        protein_feats: dict[int -> Tensor]，蛋白质原始特征
        pos_indices:   Tensor[num_classes, num_proteins]，二值矩阵
        device:        计算设备
    Returns:
        prototypes: Tensor[num_classes, hidden_dim]
    """
    model.eval()
    num_classes, num_proteins = pos_indices.shape
    # 堆叠所有蛋白质特征
    indices = sorted(protein_feats.keys(), key=int)
    all_feats = torch.stack([protein_feats[k] for k in indices], dim=0).to(device)
    all_emb = model.encode(all_feats)  # (num_proteins, hidden_dim)

    # pos_indices 和 all_emb 的蛋白质顺序需对齐
    pos = pos_indices[:, indices].float().to(device)  # (num_classes, num_proteins)
    proto_sums = pos @ all_emb                            # (num_classes, hidden_dim)
    counts = pos.sum(1).clamp(min=1)                      # (num_classes,)
    prototypes = proto_sums / counts.unsqueeze(1)
    return prototypes
