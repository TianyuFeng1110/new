import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.loss import AsymmetricLossMultiLabel

class Model(nn.Module):
    def __init__(self, prototype_feats, frequencies, input_dim, hidden_dim, num_classes, dropout=0.4, eps=1e-6):
        super(Model, self).__init__()
        
        self.prototype_feats = prototype_feats # 原型特征
        self.frequencies = frequencies
        self.eps = eps # 用于稳定数值

        self.LayerNorm = nn.LayerNorm(input_dim)

        # 原型概率可学习的参数τ和bias
        self.base_tau = 5.0 # tau的缩放因子
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.zeros(num_classes))
        # 概率结合可学习的参数: 中心阈值c和平滑系数k
        self.gate_k = nn.Parameter(torch.tensor([2.0])) 
        self.gate_c = nn.Parameter(torch.tensor([0.5]))
        # self.gate_c = torch.log10(torch.tensor(50 + 1.0))

        proj_dim = hidden_dim // 2
        self.projector = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

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
            eps=eps 
        )

    def forward(self, data, labels):
        
        protein_features = self.LayerNorm(data)
        prototype_feats = self.LayerNorm(self.prototype_feats)

        # MLP预测
        mlp_logits = self.classifier(protein_features)
        mlp_loss = self.loss_fn(mlp_logits, labels)
        probs_mlp = torch.sigmoid(mlp_logits)

        # 先投影至同一空间，再归一化：消除模长干扰 
        protein_features = F.normalize(self.projector(protein_features), p=2, dim=1)
        prototype_feats = F.normalize(self.projector(prototype_feats), p=2, dim=1)
        distances = torch.cdist(protein_features, prototype_feats, p=2.0) ** 2 # 计算欧氏距离的平方, distances[i, j] 表示第 i 个蛋白质到第 j 个原型的距离
        tau = -torch.exp(self.log_tau) * self.base_tau # tau 一定为负
        proto_logits = tau * distances + self.b
        probs_proto = torch.sigmoid(proto_logits)
        proto_loss = self.loss_fn(proto_logits, labels)

        # 联合概率
        sigma = 1.0 / (1.0 + torch.exp(-self.gate_k * (self.frequencies - self.gate_c)))
        final_probs = sigma * probs_mlp.detach() + (1.0 - sigma) * probs_proto.detach() # 截断两部分概率的计算图，要求final_loss只负责优化门控参数k和c
        # final_probs = sigma * probs_mlp + (1.0 - sigma) * probs_proto

        # 通过概率倒推逻辑上的logits
        final_logits = torch.logit(final_probs, eps=self.eps)
        gate_loss = self.loss_fn(final_logits, labels)

        return probs_mlp, probs_proto, final_probs, mlp_loss, proto_loss, gate_loss, tau, self.b, self.gate_k, self.gate_c
        