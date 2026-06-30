import torch.nn as nn
import torch
from timm.loss import AsymmetricLossMultiLabel
import torch.nn.functional as F
import copy


class Model(nn.Module):
    """功能特异原型网络：蛋白MLP + GO MLP → 哈达玛积 → 动量原型 → 余弦相似度"""

    def __init__(self, input_dim, hidden_dim, num_classes, 
                 raw_feats, queue_indices, dropout=0.4, momentum=0.999, queue_size=16):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.momentum = momentum
        self.register_buffer('raw_feats', raw_feats)
        self.register_buffer("queue_indices", queue_indices)

        self.go_feats = nn.Parameter(torch.empty(num_classes, hidden_dim))
        nn.init.normal_(self.go_feats, std=0.02)
        # 蛋白特征 MLP: (batch, input_dim) → (batch, hidden_dim)
        self.protein_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.momentum_protein_mlp = copy.deepcopy(self.protein_mlp)
        self.momentum_go_feats = copy.deepcopy(self.go_feats)
        self.momentum_protein_mlp.eval()  # 动量模型始终 eval，关闭 Dropout
        self._freeze_momentum_params()

        # 可学习温度与偏置
        self.tau = nn.Parameter(torch.ones(num_classes) * 2.0)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        
        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4, gamma_pos=0, clip=0.0, eps=1e-8,
        )

    def train(self, mode=True):
        super().train(mode)
        # 动量子模块始终保持在 eval 模式，关闭 Dropout
        self.momentum_protein_mlp.eval()
        return self

    def forward(self, input_feats, labels):

        protein_feats = self.protein_mlp(input_feats)                 # (batch, hidden_dim)

        # 2. 哈达玛积 → 功能特异蛋白特征: (batch, num_classes, hidden_dim)
        func_specific = protein_feats.unsqueeze(1) * self.go_feats.unsqueeze(0)

        # 3. 动量原型: (num_classes, hidden_dim)
        prototypes = self._compute_prototypes()

        # 4. 余弦相似度 → logits
        func_norm = F.normalize(func_specific, p=2, dim=-1)
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        cos_sim = (func_norm * proto_norm.unsqueeze(0)).sum(dim=-1)  # (batch, num_classes)
        logits = torch.exp(self.tau) * cos_sim + self.bias

        # loss = self.loss_fn(logits, labels)
        # probs = torch.sigmoid(logits)

        return logits, self.tau, self.bias

    def _freeze_momentum_params(self):
        for m in [self.momentum_protein_mlp]:
            for p in m.parameters():
                p.requires_grad = False
        self.momentum_go_feats.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        """EMA: θ_m ← m·θ_m + (1-m)·θ"""
        for src, dst in [
            (self.protein_mlp, self.momentum_protein_mlp),
        ]:
            for p_src, p_dst in zip(src.parameters(), dst.parameters()):
                p_dst.copy_(self.momentum * p_dst + (1 - self.momentum) * p_src)

        self.momentum_go_feats.data = self.momentum * self.momentum_go_feats.data + (1 - self.momentum) * self.go_feats.data

    @torch.no_grad()
    def _compute_prototypes(self):
        """动量模型 → 队列功能特异特征 → 按类平均得原型 (num_classes, hidden_dim)"""

        # 只取出队列中的蛋白质: (num_classes, queue_size, dim)
        idx = self.queue_indices.clamp(min=0)                         # -1 → 0 安全索引
        valid_mask = (self.queue_indices >= 0).unsqueeze(-1).float()  # (num_classes, queue_size, 1)

        class_num, Q_size = idx.shape
        prot_feats = self.raw_feats[idx]                            
        # 展平过 MLP 后恢复，避免构造 (data_num, num_classes, hidden_dim) 巨量张量
        prot_feats = self.momentum_protein_mlp(prot_feats.view(class_num * Q_size, -1)).view(class_num, Q_size, -1)

        # 哈达玛积
        func_all = prot_feats * self.momentum_go_feats.unsqueeze(1) * valid_mask
        proto_sums = func_all.sum(dim=1)                              # (num_classes, hidden_dim)
        counts = valid_mask.sum(dim=1).clamp(min=1.0)                 # (num_classes, 1)
        return proto_sums / counts
