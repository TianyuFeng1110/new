import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """门控融合：冻结的 MLP 分类头 + 冻结的原型网络。

    sigma = (log_freq - min_log) / (max_log - min_log)
        - 对 GO term 频率做 log 变换后 min-max 归一化到 [0, 1]
        - 零自定义参数：min/max 完全来自数据自身的 log-频率分布
        - 可解释性：sigma = 0.3 表示该功能在 log-频率谱上位于最罕见到最常见之间的 30% 位置

    final_probs = sigma * mlp_probs + (1 - sigma) * proto_probs
    高频功能 sigma→1（MLP 主导），低频功能 sigma→0（原型网络主导）。
    """

    def __init__(self, mlp_model, proto_model, go_freq):
        super().__init__()
        self.num_classes = go_freq.shape[0]

        # 两个预训练模型，冻结不参与训练
        self.mlp_model = mlp_model
        self.proto_model = proto_model
        for p in self.mlp_model.parameters():
            p.requires_grad = False
        for p in self.proto_model.parameters():
            p.requires_grad = False

        # ---- 纯数学计算 sigma：log-频率 min-max 归一化 ----
        log_freq = torch.log(go_freq + 1e-8)
        min_log = log_freq.min()
        max_log = log_freq.max()
        sigma = (log_freq - min_log) / (max_log - min_log).clamp(min=1e-8)

        self.register_buffer('sigma', sigma)
        self.register_buffer('log_freq', log_freq)

    def train(self, mode=True):
        super().train(mode)
        self.mlp_model.eval()
        self.proto_model.eval()
        return self

    def forward(self, input_feats, prototypes, labels):
        with torch.no_grad():
            # MLP 分类头
            _, custom_logits, _ = self.mlp_model(input_feats, None, labels)
            custom_probs = torch.sigmoid(custom_logits)

            # 原型网络：MLP 编码 → 与预计算原型做余弦相似度
            q_feats = self.proto_model.mlp(input_feats)
            proto_logits = self.proto_model.predict(q_feats, prototypes)
            proto_probs = torch.sigmoid(proto_logits)

        # 门控融合（sigma 固定，无梯度）
        final_probs = self.sigma * custom_probs + (1.0 - self.sigma) * proto_probs

        logits = torch.logit(final_probs, eps=1e-6)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        return final_probs, loss, self.sigma, torch.tensor(0.0), torch.tensor(0.0)

