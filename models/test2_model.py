import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """门控融合：冻结的 MLP 分类头 + 冻结的原型网络，仅训练门控参数。

    sigma = sigmoid(k * (log_freq - c))
    final_probs = sigma * mlp_probs + (1 - sigma) * proto_probs

    高频功能 sigma→1（MLP 主导），低频功能 sigma→0（原型网络主导）。
    """

    def __init__(self, mlp_model, proto_model, go_freq, num_quantiles=100, lambda_sigma=1.0):
        super().__init__()
        self.num_classes = go_freq.shape[0]
        self.num_quantiles = num_quantiles
        self.lambda_sigma = lambda_sigma

        # 两个预训练模型，冻结不参与训练
        self.mlp_model = mlp_model
        self.proto_model = proto_model
        for p in self.mlp_model.parameters():
            p.requires_grad = False
        for p in self.proto_model.parameters():
            p.requires_grad = False

        # 门控参数
        self.register_buffer('log_freq', torch.log(go_freq + 1e-6))
        self.raw_gate_k = nn.Parameter(torch.ones(self.num_classes))
        self.gate_c = nn.Parameter(self.log_freq.clone())

        # 逐类百分位归一化表（预计算后通过 set_quantiles 写入）
        self.register_buffer('mlp_quantiles', torch.zeros(self.num_classes, num_quantiles + 1))
        self.register_buffer('proto_quantiles', torch.zeros(self.num_classes, num_quantiles + 1))

        # 频率先验：罕见类→0（信 proto），高频类→1（信 MLP）
        beta = 10.0
        self.register_buffer('freq_prior',
            torch.sigmoid(beta * (self.log_freq - self.log_freq.median())))

    def set_quantiles(self, mlp_q, proto_q):
        """写入预计算的 per-class 百分位边界表。"""
        self.mlp_quantiles.copy_(mlp_q)
        self.proto_quantiles.copy_(proto_q)

    def _quantile_normalize(self, probs, quantiles):
        """将原始概率映射到 [0,1] 百分位（固定变换，不参与梯度）。每个类别中probs概率的顺序是不变的，值进行了缩放，具体可以观察custom_n和custom_probs的值

        probs:     (B, C)
        quantiles: (C, Q+1)
        """
        idx = torch.searchsorted(quantiles, probs.t().contiguous(), side='left')
        return (idx.float() / self.num_quantiles).t().clamp(0.0, 1.0)

    def train(self, mode=True):
        super().train(mode)
        # 两个子模型始终 eval，关闭 Dropout
        self.mlp_model.eval()
        self.proto_model.eval()
        return self

    def forward(self, input_feats, prototypes, labels):
        # 冻结模型前向 + 百分位归一化，不计算梯度
        with torch.no_grad():
            # MLP 分类头
            _, custom_logits, _ = self.mlp_model(input_feats, None, labels)
            custom_probs = torch.sigmoid(custom_logits)

            # 原型网络：MLP 编码 → 与预计算原型做余弦相似度
            q_feats = self.proto_model.mlp(input_feats)
            proto_logits = self.proto_model.predict(q_feats, prototypes)
            proto_probs = torch.sigmoid(proto_logits)

            # 逐类百分位归一化（仅用于 loss 计算，修 P1/P2）
            custom_n = self._quantile_normalize(custom_probs, self.mlp_quantiles)
            proto_n  = self._quantile_normalize(proto_probs,  self.proto_quantiles)

        # ---- 门控参数（同一组 sigma 供两条路径复用） ----
        gate_k = F.softplus(self.raw_gate_k)
        sigma = torch.sigmoid(gate_k * (self.log_freq - self.gate_c))

        # 训练路径：归一化分数 → loss_bce（P1 梯度无偏 + P2 排序代理）
        final_norm = sigma * custom_n + (1.0 - sigma) * proto_n
        loss_bce = F.binary_cross_entropy_with_logits(
            torch.logit(final_norm, eps=1e-6), labels)

        # 频率先验 → sigma 监督（骨架：罕见类信 proto，高频类信 MLP）
        loss_sigma = F.mse_loss(sigma, self.freq_prior)
        loss = loss_bce + self.lambda_sigma * loss_sigma

        # 评估路径：原始分数 → Fmax（P3 保留稀疏分布，全局阈值有效）
        final_raw = sigma * custom_probs + (1.0 - sigma) * proto_probs

        return final_raw, loss, sigma, gate_k.mean(), self.gate_c.mean()

