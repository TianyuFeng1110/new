import torch.nn as nn
import torch
from timm.loss import AsymmetricLossMultiLabel
import torch.nn.functional as F

class CustomModel(nn.Module):
    """简单的两层 MLP 基准模型：均值池化 + Linear-ReLU-Dropout-Linear"""

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.4):
        super(CustomModel, self).__init__()

        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim * 2, hidden_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
        )

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=0,
            clip=0.05,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=1e-8 
        )
        # self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_feats, labels):

        # 轻量级网络
        feats = self.proj(input_feats)

        custom_logits = self.classifier(feats)
        custom_loss = self.loss_fn(custom_logits, labels)
        custom_probs = torch.sigmoid(custom_logits)
        
        return custom_probs, custom_logits, custom_loss