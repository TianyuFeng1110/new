import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """门控融合：冻结的 MLP 分类头 + 冻结的原型网络。

    sigma = n / (n + tau_g)，n 为类别在训练集中的正样本数：
        - n = 0（零样本类）: sigma = 0，完全信任原型网络
          （MLP 对无监督类结构性失明，原型网络经祖先平滑仍可预测）
        - n >= 1（已见类）: sigma 随 n 迅速饱和到 1，信任 MLP
          （实验表明已见类全频段 MLP 均不弱于原型网络）
        - tau_g 为半饱和超参数：n = tau_g 时 sigma = 0.5，
          控制"监督可用性"边界的软硬程度（tau_g 越小越接近硬门控）

    final_probs = sigma * mlp_probs + (1 - sigma) * proto_probs
    门控按"监督是否存在"分工，而非频率连续谱。
    """

    def __init__(self, mlp_model, proto_model, class_counts, tau_g=2.0, eps=1e-8):
        super().__init__()
        self.num_classes = class_counts.shape[0]
        self.tau_g = tau_g

        # 两个预训练模型，冻结不参与训练
        self.mlp_model = mlp_model
        self.proto_model = proto_model
        for p in self.mlp_model.parameters():
            p.requires_grad = False
        for p in self.proto_model.parameters():
            p.requires_grad = False

        # ---- 监督可用性门控：sigma = n / (n + tau_g) ----
        n = class_counts.float()
        sigma = n / (n + tau_g + eps)   # n=0 → 0（信 Proto）；n 增大 → 1（信 MLP）,eps防止除0

        self.register_buffer('sigma', sigma)

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

        return final_probs, self.sigma

