import torch.nn as nn
import torch
from timm.loss import AsymmetricLossMultiLabel
import torch.nn.functional as F

class PrototypeModule(nn.Module):
    def __init__(self, num_classes):
        super(PrototypeModule, self).__init__()
        self.base_tau = 1.0  # 缩小缩放因子，避免初始 proto_logits 过于极端
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, pooled_feats, prototype_feats):
        protein_feats = F.normalize(pooled_feats, p=2, dim=1)
        prototype_feats = F.normalize(prototype_feats, p=2, dim=1)
        distances = torch.cdist(protein_feats, prototype_feats, p=2.0) ** 2 # 计算欧氏距离的平方, distances[i, j] 表示第 i 个蛋白质到第 j 个原型的距离
        tau = -torch.exp(self.log_tau) * self.base_tau # tau 一定为负
        proto_logits = tau * distances + self.b
        # proto_loss = self.loss_fn(proto_logits, labels)
        # proto_probs = torch.sigmoid(proto_logits)
        return proto_logits

class Model(nn.Module):
    """简单的两层 MLP 基准模型：均值池化 + Linear-ReLU-Dropout-Linear"""

    def __init__(self, input_dim, hidden_dim, num_classes, prototype_index, proto_idx_mask, go_freq, dropout=0.4):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.register_buffer('prototype_index', prototype_index)
        self.register_buffer('proto_idx_mask', proto_idx_mask)
        self.register_buffer('frequency', go_freq)

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

        # protein_feats 和 prototype_feats 在 init_protein_feats() / init_prototype_feats() 中惰性注册
        self.proto_module = PrototypeModule(num_classes)

        self.register_buffer('log_freq', torch.log(self.frequency + 1e-6))
        self.raw_gate_k = nn.Parameter(torch.tensor([1.0])) 
        self.gate_c = nn.Parameter(torch.tensor([self.log_freq.mean()]))

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=0,
            clip=0.0,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=1e-8 
        )

    def forward(self, input_feats, indices, labels):

        # 更新原型
        if self.training:
            self.prototype_feats.copy_(self._compute_prototype_feats(self.num_classes, self.hidden_dim, self._compute_protein_feats(self.raw_feats)))

        # 轻量级网络
        feats = self.proj(input_feats)

        custom_logits = self.classifier(feats)
        custom_loss = self.loss_fn(custom_logits, labels)
        custom_probs = torch.sigmoid(custom_logits)

        # 原型网络
        proto_logits = self.proto_module(feats, self.prototype_feats)
        proto_loss = self.loss_fn(proto_logits, labels)
        proto_probs = torch.sigmoid(proto_logits)

        # 融合
        gate_k = F.softplus(self.raw_gate_k)
        sigma = torch.sigmoid(gate_k * (self.log_freq - self.gate_c))# TODO self.freq_norm用于数值稳定，平均值为0，95%的数值落在-2到2之内。但是这样就gate c就要初始化为0，但是目前的初始化方法会导致gate c可能为负，导致sigma上升(0.5 ＜ sigmod(k=2 * (freq_norm=0 - gate_c=负数) ))
        final_probs = sigma * custom_probs + (1.0 - sigma) * proto_probs

        # 通过概率倒推逻辑上的logits
        logits = torch.logit(final_probs, eps=1e-6)
        gate_loss = self.loss_fn(logits, labels)
        
        return final_probs, custom_logits, custom_loss, proto_loss, gate_loss, sigma, gate_k, self.gate_c

    @torch.no_grad()
    def _compute_protein_feats(self, global_feats):
        """纯计算：投影得到蛋白质特征矩阵，该矩阵只用于计算原型，计算原型时要避免dropout等手段的干扰，使用全量模型"""
        mod = self.training
        self.eval()
        try:
            projed_feats = self.proj(global_feats)
        finally:
            self.train(mod) 
            
        return projed_feats

    def _compute_prototype_feats(self, num_classes, hidden_dim, protein_feats):
        """
        纯计算：用每个 GO term 对应蛋白质的平均特征计算 prototype_feats。
        """

        valid_mask = self.proto_idx_mask                     # (num_classes, max_proteins)
        counts = valid_mask.sum(dim=1)                        # (num_classes,)

        prototype_feats = torch.randn(
            num_classes, hidden_dim,
            device=protein_feats.device, dtype=protein_feats.dtype
        )

        nnz = int(counts.sum().item())
        if nnz == 0:
            return prototype_feats

        nonzero_rows = valid_mask.nonzero(as_tuple=False)     # (nnz, 2)
        class_ids = nonzero_rows[:, 0]                        # (nnz,)
        protein_ids = self.prototype_index[valid_mask]        # (nnz,)  布尔索引与 nonzero 同序

        summed = torch.zeros(
            num_classes, hidden_dim,
            device=protein_feats.device, dtype=protein_feats.dtype
        )
        summed.index_add_(0, class_ids, protein_feats[protein_ids])  # (nnz, hidden_dim)

        has = counts > 0
        prototype_feats[has] = summed[has] / counts[has].unsqueeze(-1)

        return prototype_feats

    @torch.no_grad()
    def init_prototype_feats(self, num_classes, hidden_dim, raw_feats):
        """初始化时调用：eval 模式下无梯度计算并注册 buffer"""
        self.eval()
        self.register_buffer('raw_feats', raw_feats)
        self.register_buffer('prototype_feats', self._compute_prototype_feats(num_classes, hidden_dim, self._compute_protein_feats(self.raw_feats)))
        self.train()

