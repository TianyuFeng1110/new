import torch.nn as nn
import torch
from timm.loss import AsymmetricLossMultiLabel

class Model(nn.Module):
    """简单的两层 MLP 基准模型：均值池化 + Linear-ReLU-Dropout-Linear"""

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.4):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
        )

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=0,
            clip=0.0,
            eps=1e-8 
        )

    def forward(self, input_feats, indices, labels):
        feats = self.proj(input_feats)
        logits = self.classifier(feats)
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        return probs, logits, loss

