import torch.nn as nn
import torch
from timm.loss import AsymmetricLossMultiLabel
import torch.nn.functional as F
import copy


class Model(nn.Module):
    """轻量级 MLP 分类头 + 功能特异原型网络（哈达玛积→动量原型→余弦相似度）+ 门控融合"""

    def __init__(self, input_dim, hidden_dim, num_classes, go_freq,
                 raw_feats, go_embeddings, queue_indices, dropout=0.4, momentum=0.999, queue_size=16):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.momentum = momentum

        self.register_buffer('frequency', go_freq)
        self.register_buffer('raw_feats', raw_feats)
        self.register_buffer("queue_indices", queue_indices)

        # GO 嵌入: (num_classes, go_emb_dim)，可切换为可学习参数
        go_emb_dim = go_embeddings.shape[1]
        self.register_buffer('go_embeddings', go_embeddings)
        # self.go_embeddings = nn.Parameter(go_embeddings)  # NOTE 改解冻 go embedding 时放开该行，注释上行

        # ====== 轻量级 MLP + 分类头（保持不变） ======
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

        # ====== 新原型网络：蛋白 MLP + GO MLP → 哈达玛积 → 动量原型 → 余弦相似度 ======
        # 蛋白特征 MLP: (batch, input_dim) → (batch, hidden_dim)
        self.protein_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GO 特征 MLP: (num_classes, go_emb_dim) → (num_classes, hidden_dim)
        self.go_mlp = nn.Sequential(
            nn.LayerNorm(go_emb_dim),
            nn.Linear(go_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.class_specific_residual = nn.Parameter(torch.zeros(num_classes, hidden_dim))

        # 动量模型（EMA），始终 eval 关闭 Dropout
        self.momentum_protein_mlp = copy.deepcopy(self.protein_mlp)
        self.momentum_go_mlp = copy.deepcopy(self.go_mlp)
        self.momentum_protein_mlp.eval()
        self.momentum_go_mlp.eval()
        self._freeze_momentum_params()

        # 可学习温度与偏置（原型网络）
        self.log_tau = nn.Parameter(torch.ones(num_classes) * 2.0)
        self.bias = nn.Parameter(torch.zeros(num_classes))

        # ====== 门控融合：per-class 参数，每类独立学习 ======
        self.register_buffer('log_freq', torch.log(self.frequency + 1e-6))
        self.raw_gate_k = nn.Parameter(torch.ones(num_classes))       # (num_classes,)
        self.gate_c = nn.Parameter(self.log_freq.mean())  # (num_classes,)

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=0,
            clip=0.0,
            eps=1e-8
        )

    def train(self, mode=True):
        super().train(mode)
        # 动量子模块始终保持在 eval 模式，关闭 Dropout
        self.momentum_protein_mlp.eval()
        self.momentum_go_mlp.eval()
        return self

    def forward(self, input_feats, indices, labels):

        # ====== 轻量级网络（保持不变） ======
        feats = self.proj(input_feats)
        custom_logits = self.classifier(feats)
        custom_probs = torch.sigmoid(custom_logits)

        # ====== 新原型网络：哈达玛积 → 动量原型 → 余弦相似度 ======
        go_feats = self.go_mlp(self.go_embeddings)                     # (num_classes, hidden_dim)
        go_feats = go_feats + self.class_specific_residual
        protein_feats = self.protein_mlp(input_feats)                  # (batch, hidden_dim)

        # 哈达玛积 → 功能特异蛋白特征: (batch, num_classes, hidden_dim)
        func_specific = protein_feats.unsqueeze(1) * go_feats.unsqueeze(0)

        # 动量原型: (num_classes, hidden_dim)
        prototypes = self._compute_prototypes()

        # 余弦相似度 → logits
        func_norm = F.normalize(func_specific, p=2, dim=-1)
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        cos_sim = (func_norm * proto_norm.unsqueeze(0)).sum(dim=-1)    # (batch, num_classes)
        proto_logits = torch.exp(self.log_tau) * cos_sim + self.bias
        proto_probs = torch.sigmoid(proto_logits)

        # ====== 门控融合：per-class 独立门控 ======
        gate_k = F.softplus(self.raw_gate_k)                         # (num_classes,)
        sigma = torch.sigmoid(gate_k * (self.log_freq - self.gate_c))  # (num_classes,)
        final_probs = sigma * custom_probs + (1.0 - sigma) * proto_probs

        # 通过概率倒推逻辑上的 logits
        logits = torch.logit(final_probs, eps=1e-6)
        gate_loss = self.loss_fn(logits, labels)

        # 返回均值用于日志记录
        return final_probs, custom_logits, torch.zeros_like(gate_loss), torch.zeros_like(gate_loss), gate_loss, sigma, gate_k.mean(), self.gate_c.mean()

    def _freeze_momentum_params(self):
        for m in [self.momentum_protein_mlp, self.momentum_go_mlp]:
            for p in m.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        """EMA: θ_m ← m·θ_m + (1-m)·θ，由外部训练循环在 optimizer.step() 后调用"""
        for src, dst in [
            (self.protein_mlp, self.momentum_protein_mlp),
            (self.go_mlp, self.momentum_go_mlp),
        ]:
            for p_src, p_dst in zip(src.parameters(), dst.parameters()):
                p_dst.copy_(self.momentum * p_dst + (1 - self.momentum) * p_src)

    @torch.no_grad()
    def _compute_prototypes(self):
        """动量模型 → 队列功能特异特征 → 按类平均得原型 (num_classes, hidden_dim)"""
        go_feats = self.momentum_go_mlp(self.go_embeddings)                     # (num_classes, hidden_dim)

        # 只取出队列中的蛋白质: (num_classes, queue_size, dim)
        idx = self.queue_indices.clamp(min=0)                         # -1 → 0 安全索引
        valid_mask = (self.queue_indices >= 0).unsqueeze(-1).float()  # (num_classes, queue_size, 1)

        class_num, Q_size = idx.shape
        prot_feats = self.raw_feats[idx]
        # 展平过 MLP 后恢复，避免构造 (data_num, num_classes, hidden_dim) 巨量张量
        prot_feats = self.momentum_protein_mlp(prot_feats.view(class_num * Q_size, -1)).view(class_num, Q_size, -1)

        # 哈达玛积
        func_all = prot_feats * go_feats.unsqueeze(1) * valid_mask
        proto_sums = func_all.sum(dim=1)                              # (num_classes, hidden_dim)
        counts = valid_mask.sum(dim=1).clamp(min=1.0)                 # (num_classes, 1)
        return proto_sums / counts

