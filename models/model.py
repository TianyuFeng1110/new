import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.loss import AsymmetricLossMultiLabel

class LocalMotifFeatureExtractionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, dilations=None, dropout=0.4):
        """
        Args:
            dilations: 3层卷积各自的膨胀率，默认 [1, 1, 1]（等价于普通卷积）。
                       膨胀卷积可在不增加参数量的前提下指数级扩大感受野。
                       
            第i层的卷积核为 (k-1) * dilations_i + 1。第i层感受野为i-1层的感受野加上当前层卷积核大小再减去1。
        """
        super(LocalMotifFeatureExtractionModule, self).__init__()
        if dilations is None:
            dilations = [1, 1, 1]
        d1, d2, d3 = dilations

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # padding 根据 dilation 自动计算以保持序列长度不变: pad = (k-1) * d / 2
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                               padding=(kernel_size - 1) * d1 // 2, dilation=d1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                               padding=(kernel_size - 1) * d2 // 2, dilation=d2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                               padding=(kernel_size - 1) * d3 // 2, dilation=d3)
        # GN代替LN，以避免频繁的维度顺序切换造成的开销
        self.norm1 = nn.GroupNorm(1, hidden_dim)
        self.norm2 = nn.GroupNorm(1, hidden_dim)
        self.norm3 = nn.GroupNorm(1, hidden_dim)
        
        # 卷积层之间以及拼接后的 Dropout
        self.inter_conv_dropout = nn.Dropout(dropout)
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.LayerNorm = nn.LayerNorm(hidden_dim)

    def forward(self, residue_feats, mask):

        x = self.mlp(residue_feats)
        x = x.transpose(1, 2)
        mask_expanded = mask.unsqueeze(1)
        
        # 三层卷积堆叠处理, 卷积后乘以 mask 来屏蔽 padding 的影响
        x_conv1 = self.conv1(x)
        conv1_out = F.gelu(self.norm1(x_conv1)) * mask_expanded
        conv1_out = self.inter_conv_dropout(conv1_out)

        x_conv2 = self.conv2(conv1_out)
        conv2_out = F.gelu(self.norm2(x_conv2)) * mask_expanded
        conv2_out = self.inter_conv_dropout(conv2_out)
        
        x_conv3 = self.conv3(conv2_out)
        conv3_out = F.gelu(self.norm3(x_conv3)) * mask_expanded

        # 在 hidden_dim 维度直接拼接并降维
        merged = torch.cat([x, conv1_out, conv2_out, conv3_out], dim=1).transpose(1, 2)
        merged = self.feature_fusion(merged) * mask.unsqueeze(-1)
        merged = self.LayerNorm(merged)
        
        return merged

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, kernel_size, go_freq, dilations=None, dropout=0.4):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.residue_LayerNorm = nn.LayerNorm(input_dim)
        self.go_freq = go_freq

        self.local_feats_extraction_module = LocalMotifFeatureExtractionModule(input_dim, hidden_dim, kernel_size=kernel_size, dilations=dilations, dropout=dropout)

        # 注意力分数
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.pre_classifier_dropout = nn.Dropout(dropout)

        # 原型概率可学习的参数τ和bias
        self.base_tau = 5.0 # tau的缩放因子
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.zeros(num_classes))
        self.prototype_feats = nn.Parameter(torch.randn(num_classes, hidden_dim))

        # 概率结合可学习的参数: 中心阈值c和平滑系数k
        self.gate_k = nn.Parameter(torch.tensor([2.0])) 
        go_freq_log = torch.log(self.go_freq + 1e-6) 
        self.feq_log_mean = go_freq_log.mean()
        self.feq_log_std = go_freq_log.std()
        self.go_freq_norm = (go_freq_log - self.feq_log_mean) / (self.feq_log_std + 1e-8) # z-score标准化
        self.gate_c = nn.Parameter(torch.tensor([0.0]))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes), 
        )

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=0,
            clip=0.0,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=1e-8 
        )

    def forward(self, residue_feats_data, mask, labels):

        residue_feats_data = self.residue_LayerNorm(residue_feats_data)

        # custom 模块
        merged_feats = self.local_feats_extraction_module(residue_feats_data, mask)
        mean_feats = self._get_mean_pooled_features(merged_feats, mask)
        max_feats = self._get_max_pooled_features(merged_feats, mask)
        attn_feats = self._get_attention_pooled_features(merged_feats, mask)

        merged = torch.cat([mean_feats, max_feats, attn_feats], dim=-1) 
        merged = self.pre_classifier_dropout(merged)
        custom_logits = self.classifier(merged) # (B, num_classes)
        custom_loss = self.loss_fn(custom_logits, labels)
        custom_probs = torch.sigmoid(custom_logits)

        # 原型部分
        protein_feats = F.normalize(merged, p=2, dim=1)
        prototype_feats = F.normalize(self.prototype_feats, p=2, dim=1)
        distances = torch.cdist(protein_feats, prototype_feats, p=2.0) ** 2 # 计算欧氏距离的平方, distances[i, j] 表示第 i 个蛋白质到第 j 个原型的距离
        tau = -torch.exp(self.log_tau) * self.base_tau # tau 一定为负
        proto_logits = tau * distances + self.b
        proto_loss = self.loss_fn(proto_logits, labels)
        proto_probs = torch.sigmoid(proto_logits)

        # 理论上sigma初始为0.5
        sigma = 1.0 / (1.0 + torch.exp(-self.gate_k * (self.go_freq_norm - self.gate_c)))
        final_probs = sigma * custom_probs + (1.0 - sigma) * proto_probs

        # 通过概率倒推逻辑上的logits
        logits = torch.logit(final_probs, eps=1e-6)
        gate_loss = self.loss_fn(logits, labels)
        
        # 计算日志输出的变量(当sigma为0.5时，对应的go term频率是多少)
        target_log = self.gate_c.item() * (self.feq_log_std + 1e-8) + self.feq_log_mean # 逆 Z-Score 标准化
        target_freq = torch.exp(target_log) - 1e-6 # 逆对数变换（解出原始频率）
        target_freq = torch.clamp(target_freq, min=0.0)

        return logits, custom_logits, custom_loss, proto_loss, gate_loss, sigma.mean(), target_freq
    
    def _get_mean_pooled_features(self, merged, mask):
        # 平均池化
        mask_expanded = mask.unsqueeze(-1)
        masked_feats = merged * mask_expanded
        pooled = masked_feats.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return pooled
    
    def _get_max_pooled_features(self, merged, mask):
        # 处理 Mask：将 Padding 部分设为极小值以忽略其对 Max Pooling 的影响
        mask_expanded = mask.unsqueeze(-1)
        merged = merged.masked_fill(mask_expanded == 0, -1e9)
        
        # 全局最大池化
        pooled = torch.max(merged, dim=1)[0]
        return pooled

    def _get_attention_pooled_features(self, merged, mask):

        # 计算注意力得分
        scores = self.attn_score(merged)  # (B, L, 1)
        
        # 将 padding 位置的得分设为 -inf，使 softmax 后权重为 0
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        
        # Softmax 归一化 + 加权求和
        attn_weights = F.softmax(scores, dim=1)  # (B, L, 1)
        pooled = (merged * attn_weights).sum(dim=1)  # (B, D)
        
        return pooled

