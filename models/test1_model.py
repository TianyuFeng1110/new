import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import AsymmetricLossMultiLabel


class Model(nn.Module):
    """门控融合：冻结的 MLP 分类头 + 冻结的原型网络，仅训练门控参数。

    sigma = sigmoid(k * (log_freq - c))
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

        # 门控参数
        self.register_buffer('log_freq', torch.log(go_freq + 1e-6))
        self.raw_gate_k = nn.Parameter(torch.ones(self.num_classes))
        self.gate_c = nn.Parameter(self.log_freq.clone())

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4, gamma_pos=0, clip=0.0, eps=1e-8
        )

    def train(self, mode=True):
        super().train(mode)
        # 两个子模型始终 eval，关闭 Dropout
        self.mlp_model.eval()
        self.proto_model.eval()
        return self

    def forward(self, input_feats, prototypes, labels):
        # 冻结模型前向，不计算梯度
        with torch.no_grad():
            # MLP 分类头
            _, custom_logits, _ = self.mlp_model(input_feats, None, labels)
            custom_probs = torch.sigmoid(custom_logits)

            # 原型网络：MLP 编码 → 与预计算原型做余弦相似度
            q_feats = self.proto_model.mlp(input_feats)
            proto_logits = self.proto_model.predict(q_feats, prototypes)
            proto_probs = torch.sigmoid(proto_logits)

        # 门控融合（梯度仅流经 sigma → raw_gate_k, gate_c）
        gate_k = F.softplus(self.raw_gate_k)
        sigma = torch.sigmoid(gate_k * (self.log_freq - self.gate_c))
        final_probs = sigma * custom_probs + (1.0 - sigma) * proto_probs

        # 概率倒推 logits 计算损失
        logits = torch.logit(final_probs, eps=1e-6)
        # loss = self.loss_fn(logits, labels)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        return final_probs, loss, sigma, gate_k.mean(), self.gate_c.mean()

