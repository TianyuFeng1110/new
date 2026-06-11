import torch.nn as nn
import torch
from timm.loss import AsymmetricLossMultiLabel

class Model(nn.Module):
    """简单的两层 MLP 基准模型：均值池化 + Linear-ReLU-Dropout-Linear"""

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.4):
        super(Model, self).__init__()

        self.residue_LayerNorm = nn.LayerNorm(input_dim)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=0,
            clip=0.0,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=1e-8 
        )

    def forward(self, residue_feats_data, mask, labels):

        residue_feats_data = self.residue_LayerNorm(residue_feats_data)

        # 平均池化：沿序列维度取均值，忽略 padding 位置
        mask_expanded = mask.unsqueeze(-1).float()
        masked_feats = residue_feats_data * mask_expanded
        sum_feats = masked_feats.sum(dim=1)
        valid_counts = mask_expanded.sum(dim=1).clamp(min=1)
        pooled = sum_feats / valid_counts
        
        # # 最大池化：沿序列维度取最大值，忽略 padding 位置
        # mask_expanded = mask.unsqueeze(-1).bool()
        # # 将 padding 位置设为 -inf，这样 max 操作会忽略它们
        # masked_feats = residue_feats_data.masked_fill(~mask_expanded, float('-inf'))
        # pooled = masked_feats.max(dim=1).values
        
        logits = self.mlp(pooled)
        loss = self.loss_fn(logits, labels)
        
        return logits, loss

