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

    def __init__(self, input_dim, hidden_dim, num_classes, prototype_index, proto_idx_mask, go_freq, dropout=0.4, momentum_factor=0.99):
        super(Model, self).__init__()

        self.register_buffer('prototype_index', prototype_index)
        self.register_buffer('proto_idx_mask', proto_idx_mask)
        self.register_buffer('frequency', go_freq)
        self.momentum_factor = momentum_factor

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

        self.gate_k = nn.Parameter(torch.tensor([2.0])) 
        self.gate_c = nn.Parameter(torch.tensor([0.0]))

        # 用于初始化中心阈值
        freq_log = torch.log(self.frequency + 1e-6)
        freq_log_mean = freq_log.mean()
        freq_log_std = freq_log.std()
        freq_norm = (freq_log - freq_log_mean) / (freq_log_std + 1e-8)  # z-score标准化
        self.register_buffer('freq_log_mean', freq_log_mean)
        self.register_buffer('freq_log_std', freq_log_std)
        self.register_buffer('freq_norm', freq_norm)

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=0,
            clip=0.0,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=1e-8 
        )

    def forward(self, input_feats, indices, labels):

        # 轻量级网络
        feats = self.proj(input_feats)

        custom_logits = self.classifier(feats)
        custom_loss = self.loss_fn(custom_logits, labels)
        custom_probs = torch.sigmoid(custom_logits)

        # 原型网络
        proto_logits = self.proto_module(feats, self.prototype_feats)
        proto_loss = self.loss_fn(proto_logits, labels)
        proto_probs = torch.sigmoid(proto_logits)
        # 更新原型
        if self.training:
            self.protein_feats[indices] = feats.detach() # 更新蛋白质矩阵（detach 断开计算图，避免图累积）
            active_mask = (labels == 1).any(dim=0)
            unique_indices = torch.nonzero(active_mask, as_tuple=True)[0] # 待更新的原型索引
            self._update_prototype_feats(unique_indices)

        # 融合
        sigma = 1.0 / (1.0 + torch.exp(-self.gate_k * (self.freq_norm - self.gate_c))) # TODO self.freq_norm用于数值稳定，平均值为0，73%的数值落在-2到2之内。但是这样就gate c就要初始化为0，但是目前的初始化方法会导致gate c可能为负，导致sigma上升(0.5 ＜ sigmod(k=2 * (freq_norm=0 - gate_c=负数) ))
        final_probs = sigma * custom_probs + (1.0 - sigma) * proto_probs

        # 通过概率倒推逻辑上的logits
        logits = torch.logit(final_probs, eps=1e-6)
        gate_loss = self.loss_fn(logits, labels)
        
        return final_probs, custom_logits, custom_loss, proto_loss, gate_loss, sigma

    @torch.no_grad()
    def init_protein_feats(self, stacked_feats):
        """
        注册预池化的蛋白质特征矩阵。
        应在模型 .to(device) 之后调用，充分利用 GPU 加速。

        Args:
            pooled_feats: list of Tensors, 每个 Tensor 形状为 (input_dim,)，已在 CPU 上
            device: 目标设备（如 'cuda:0'）
        """
        self.eval()
        protein_feats = self.proj(stacked_feats)  # (N, hidden_dim)
        self.register_buffer('protein_feats', protein_feats)
        self.train()

    @torch.no_grad()
    def init_prototype_feats(self, num_classes, hidden_dim):
        """
        用每个 GO term 对应蛋白质的平均特征初始化 prototype_feats，
        避免纯随机初始化导致原型网络从一开始就失效。
        需要在 init_protein_feats() 之后调用，且模型已在目标设备上。
        """
        self.eval()
        prototype_feats = torch.randn(num_classes, hidden_dim, device=self.protein_feats.device)
        for c in range(num_classes):
            # 获取该 GO term 对应蛋白质的有效索引
            mask = self.proto_idx_mask[c]       # (max_proteins,)
            indices = self.prototype_index[c]   # (max_proteins,)
            valid_indices = indices[mask]       # 有效蛋白质索引
            
            if valid_indices.size(0) > 0:
                # 取这些蛋白质特征的平均值
                proto_mean = self.protein_feats[valid_indices].mean(dim=0)
                prototype_feats[c] = proto_mean
        
        self.register_buffer('prototype_feats', prototype_feats)
        self.train()

    def _update_prototype_feats(self, unique_indices):
        # 向量化更新 prototype_feats：只 gather 有效条目，避免 (k, data_num, dim) 中间张量
        k = unique_indices.size(0)
        device = self.protein_feats.device

        selected_indices = self.prototype_index[unique_indices]   # (k, data_num)
        selected_mask = self.proto_idx_mask[unique_indices]       # (k, data_num)

        # 构建 prototype ID 矩阵：每行填满其 prototype 序号 0..k-1
        proto_ids = torch.arange(k, device=device).unsqueeze(1).expand_as(selected_mask)

        # 只保留有效条目，展平为 1D
        flat_proto_ids = proto_ids[selected_mask]                  # (total_valid,)
        flat_indices = selected_indices[selected_mask]             # (total_valid,)

        # 一次性 gather 所有有效蛋白质的特征
        valid_feats = self.protein_feats[flat_indices]             # (total_valid, dim)

        # index_add 按 prototype 聚合并求和（纯 GPU 算子，无 Python 循环）
        sum_feats = torch.zeros(k, valid_feats.size(1), device=device, dtype=valid_feats.dtype)
        sum_feats.index_add_(0, flat_proto_ids, valid_feats)      # (k, dim)

        # 每个 prototype 的有效蛋白质数量
        valid_counts = selected_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (k, 1)

        # self.prototype_feats[unique_indices] = sum_feats / valid_counts # 直接赋值，硬更新
        # 动量更新
        new_proto = sum_feats / valid_counts
        self.prototype_feats[unique_indices] = (self.momentum_factor * self.prototype_feats[unique_indices] + (1 - self.momentum_factor) * new_proto)

    @torch.no_grad()
    def diagnose_proto_vs_custom(self, input_feats, indices, labels, num_bins=5):
        """
        按频率分桶，比较原型网络 vs MLP 在各桶的 per-class loss。
        调用方式（验证时）:
            diag = model.diagnose_proto_vs_custom(feats, indices, labels)
            for k, v in diag.items():
                print(f"{k}: custom_loss={v['custom_loss']:.4f}, proto_loss={v['proto_loss']:.4f}, "
                      f"num_terms={v['num_terms']}, sigma_mean={v['sigma_mean']:.4f}")
        """
        was_training = self.training
        self.eval()

        feats = self.proj(input_feats)
        custom_logits = self.classifier(feats)
        proto_logits = self.proto_module(feats, self.prototype_feats)

        sigma = torch.sigmoid(self.gate_k * (self.freq_norm - self.gate_c))  # (num_classes,)

        # 按 freq_norm 分桶
        freq_norm = self.freq_norm  # (num_classes,)
        bin_edges = torch.linspace(freq_norm.min(), freq_norm.max(), num_bins + 1, device=freq_norm.device)
        bin_ids = torch.bucketize(freq_norm, bin_edges[1:-1])  # 0 ~ num_bins-1

        result = {}
        for b in range(num_bins):
            mask = (bin_ids == b)
            n_terms = mask.sum().item()
            if n_terms == 0:
                continue

            # 只取该桶的 class 维度
            custom_loss_bin = self.loss_fn(custom_logits[:, mask], labels[:, mask]).item()
            proto_loss_bin = self.loss_fn(proto_logits[:, mask], labels[:, mask]).item()

            freq_range = (bin_edges[b].item(), bin_edges[b + 1].item())
            sigma_mean = sigma[mask].mean().item()

            result[f"bin_{b}_freq_[{freq_range[0]:.2f},{freq_range[1]:.2f})"] = {
                "custom_loss": custom_loss_bin,
                "proto_loss": proto_loss_bin,
                "num_terms": n_terms,
                "sigma_mean": sigma_mean,
            }

        if was_training:
            self.train()
        return result