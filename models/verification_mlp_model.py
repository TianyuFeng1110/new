import torch.nn as nn
from timm.loss import AsymmetricLossMultiLabel

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.4, eps=1e-6):
        super(Model, self).__init__()
        
        self.LayerNorm = nn.LayerNorm(input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.05,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=eps   # 用于稳定数值
        )

    def forward(self, data, labels):
        
        protein_features = self.LayerNorm(data)

        logits = self.classifier(protein_features)
        loss = self.loss_fn(logits, labels)

        return logits, loss
        