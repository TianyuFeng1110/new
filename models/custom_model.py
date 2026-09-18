import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import AsymmetricLossMultiLabel


def _inv_softplus(x):
    """返回满足 softplus(y) = x 的 y (x > 0)，用于初始化恒正的可学习参数。"""
    x = float(x)
    if x > 20.0:
        return x  # x 较大时 softplus(x) ≈ x
    return math.log(math.expm1(x))


def _inv_sigmoid(p):
    """返回满足 sigmoid(y) = p 的 y (0 < p < 1)，用于初始化残差系数 β。"""
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class Model(nn.Module):
    """两层 MLP 基准模型：均值池化 + Linear-ReLU-Dropout-Linear。

    分类头支持两种可选配置 (use_ancestor_weighting):
    - False: 经典 MLP 分类头 (nn.Linear)，作为消融实验基准，与原始实现完全一致。
    - True:  "自由残差 + 祖先加权构造" 的层级感知分类头 (true-path 规则软约束)。
             将分类头对类别 i 的参数分解为自由项与祖先构造项:
                 w_i = w_free_i + β · Σ_{j∈Anc(i)} α̂_ij · w_free_j
                 b_i = b_free_i + β · Σ_{j∈Anc(i)} α̂_ij · b_free_j
             其中祖先权重与 prototype_model 的 proto_w 计算逻辑一致
             (proto_w 乘祖先的均值原型特征，此处 α 乘祖先的自由分类头参数):
                 α_ij = exp(-γ · max(IC_i - IC_j, 0)) · n_j / (n_j + λ)
             α̂ 为 α 的行归一化版本 (对祖先加权平均并归一化)；
             β = sigmoid(raw_beta) ∈ (0,1) 为可学习标量，控制层级约束强度。
             根节点 (无祖先) 的构造项为 0，自动回退为自由参数。
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.4,
                 use_ancestor_weighting=True, ancestor_mask=None, ic=None,
                 class_counts=None, alpha_lambda_init=10.0, ic_gamma_init=2.0,
                 beta_init=0.5):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_ancestor_weighting = use_ancestor_weighting

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

        if use_ancestor_weighting:
            assert ancestor_mask is not None and ic is not None and class_counts is not None, \
                '祖先加权分类头需要传入 ancestor_mask / ic / class_counts'
            # 层级先验缓冲区 (由训练集 DAG 与标注统计得到，不参与训练)
            self.register_buffer('ancestor_mask', torch.as_tensor(ancestor_mask, dtype=torch.bool))
            self.register_buffer('ic', torch.as_tensor(ic, dtype=torch.float32))
            self.register_buffer('class_counts', torch.as_tensor(class_counts, dtype=torch.float32))
            # α 的可学习超参: softplus 重参数化保证恒正 (与原型网络一致)，配置值仅作为初始值
            # λ (alpha_lambda): 样本量敏感度;  γ (ic_gamma): IC 差衰减速率
            self.raw_alpha_lambda = nn.Parameter(torch.tensor(_inv_softplus(alpha_lambda_init), dtype=torch.float32))
            self.raw_ic_gamma = nn.Parameter(torch.tensor(_inv_softplus(ic_gamma_init), dtype=torch.float32))
            # 残差系数 β ∈ (0,1): sigmoid 重参数化保证数值稳定且有界
            self.raw_beta = nn.Parameter(torch.tensor(_inv_sigmoid(beta_init), dtype=torch.float32))
            # 自由参数 w_free / b_free，形状与原 nn.Linear 的 weight / bias 一致，
            # 初始化方式也与 nn.Linear 默认初始化保持一致
            self.w_free = nn.Parameter(torch.empty(num_classes, hidden_dim))
            self.b_free = nn.Parameter(torch.empty(num_classes))
            nn.init.kaiming_uniform_(self.w_free, a=math.sqrt(5))
            bound = 1.0 / math.sqrt(hidden_dim)
            nn.init.uniform_(self.b_free, -bound, bound)
        else:
            # 经典 MLP 分类头 (消融基准)，保持原始实现
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, num_classes),
            )

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=0,
            clip=0.0,
            eps=1e-8
        )

    def _compute_ancestor_alpha(self):
        """计算祖先权重矩阵 α，逻辑与 prototype_model._compute_proto_w 一致。

        α[i][j] = exp(-γ · max(IC_i - IC_j, 0)) · n_j / (n_j + λ)，j ∈ Anc(i):
        - ΔIC = IC_i - IC_j: 惩罚过于泛化的祖先，可学习 γ 控制衰减速率;
        - n_j / (n_j + λ): 抑制样本量过小的罕见祖先，可学习 λ 调节样本量敏感度;
        - 非祖先位置为 0。

        Returns:
            alpha: (num_classes, num_classes) 祖先权重矩阵，alpha[i][j] 为祖先 j 相对功能 i 的权重
        """
        lam = F.softplus(self.raw_alpha_lambda)   # 可学习 λ > 0
        gamma = F.softplus(self.raw_ic_gamma)     # 可学习 γ > 0

        # ΔIC[i, j] = IC_i - IC_j
        ic_diff = self.ic.unsqueeze(1) - self.ic.unsqueeze(0)        # (C, C)
        ic_factor = torch.exp(-gamma * ic_diff.clamp(min=0.0))

        n = self.class_counts                                        # (C,)
        sample_factor = n / (n + lam)                                # 样本量缩放因子（按列广播，对应祖先 j）

        alpha = ic_factor * sample_factor.unsqueeze(0)
        alpha = alpha * self.ancestor_mask.float()                   # 非祖先位置置 0
        return alpha

    def _effective_classifier(self):
        """按"自由残差 + 祖先加权构造"计算有效分类头参数。

        w = w_free + β · A_weight,  b = b_free + β · A_bias
        A_weight_i = Σ_{j∈Anc(i)} α̂_ij · w_free_j，bias 同理;
        α̂ 为 α 的行归一化版本 (对祖先加权平均并归一化，与原型网络的 proto_w 用法一致)。
        根节点 (无祖先) 的 α 行全为 0，构造项为 0，自动回退为自由参数。

        Returns:
            (weight, bias): 有效分类头参数，形状 (num_classes, hidden_dim) 与 (num_classes,)
        """
        alpha = self._compute_ancestor_alpha()                       # (C, C)
        alpha_sum = alpha.sum(dim=1, keepdim=True)                   # (C, 1)，根节点为 0
        alpha_norm = alpha / alpha_sum.clamp(min=1e-8)               # 行归一化 → 祖先加权平均
        beta = torch.sigmoid(self.raw_beta)                          # 可学习 β ∈ (0, 1)

        a_weight = alpha_norm @ self.w_free                          # (C, D)
        a_bias = alpha_norm @ self.b_free                            # (C,) 矩阵-向量积
        weight = self.w_free + beta * a_weight
        bias = self.b_free + beta * a_bias                           # (C,)
        return weight, bias

    def get_ancestor_params(self):
        """返回 (lambda, gamma, beta) 的当前实际值 (float)，用于日志打印。"""
        lam = F.softplus(self.raw_alpha_lambda).item()
        gamma = F.softplus(self.raw_ic_gamma).item()
        beta = torch.sigmoid(self.raw_beta).item()
        return lam, gamma, beta

    def forward(self, input_feats, indices, labels):
        feats = self.proj(input_feats)
        if self.use_ancestor_weighting:
            weight, bias = self._effective_classifier()
            logits = F.linear(feats, weight, bias)
        else:
            logits = self.classifier(feats)
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        return probs, logits, loss
