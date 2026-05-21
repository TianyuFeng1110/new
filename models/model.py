import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.loss import AsymmetricLossMultiLabel

class CrossAttentionModule(nn.Module):
    def __init__(self, go_dim, input_dim, hidden_dim, num_classes, num_heads=8, dropout=0.3, eps=1e-5):
        super(CrossAttentionModule, self).__init__()

        self.residue_proj = nn.Sequential(
                                nn.Linear(input_dim, hidden_dim),
                                nn.LayerNorm(hidden_dim),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.LayerNorm(hidden_dim)
                            )
        self.go_proj = nn.Sequential(
                                nn.Linear(go_dim, hidden_dim),
                                nn.LayerNorm(hidden_dim),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.LayerNorm(hidden_dim)
                            )

        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                         num_heads=num_heads, 
                                         dropout=dropout, 
                                         batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.05,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=eps 
        )

        
    def forward(self, go_features, residue_feats_data, mask, labels):
        """
        go_features: [Batch, Num_GO, Dim]
        residue_feats_data: [Batch, Residue_Len, Dim]
        residue_mask: [Batch, Residue_Len] (1 表示有效残基, 0 表示 padding)
        
        返回: [Batch, Num_GO, Dim] 的特征矩阵
        """
        # 缩放维度
        go_features = self.go_proj(go_features)
        residue_feats_data = self.residue_proj(residue_feats_data)

        batch_size = residue_feats_data.size(0)
        go_features = go_features.unsqueeze(0).expand(batch_size, -1, -1) 
        key_padding_mask = (mask == 0) # nn.MultiheadAttention 的 key_padding_mask 规则：True 表示忽略 (mask out)

        attn_output, _ = self.mha(
            query=go_features,
            key=residue_feats_data,
            value=residue_feats_data,
            key_padding_mask=key_padding_mask
        )

        # 残差连接与层归一化
        output = self.layer_norm(attn_output) + go_features

        # 得到最终 Logits
        logits = self.classifier(output).squeeze(-1)
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        return logits, probs, loss

class MaxFeatureModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.4, eps=1e-5):
        super(MaxFeatureModule, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim)
        )

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)
        
        self.LayerNorm = nn.LayerNorm(hidden_dim * 3)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, num_classes)
        )

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.05,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=eps 
        )

    def forward(self, residue_feats, mask, labels):

        x = self.mlp(residue_feats)
        # 输入维度转换: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # 三路卷积并行处理
        x1 = F.gelu(self.conv1(x))
        x2 = F.gelu(self.conv2(x))
        x3 = F.gelu(self.conv3(x))

        # 在通道(channel)维度拼接: (batch, hidden_dim * 3, seq_len)
        merged = torch.cat([x1, x2, x3], dim=1)
        merged = self.LayerNorm(merged.transpose(1, 2)).transpose(1, 2)
        
        # 处理 Mask：将 Padding 部分设为极小值以忽略其对 Max Pooling 的影响
        if mask is not None:
            # mask 形状 (batch, seq_len) 扩展为 (batch, 1, seq_len)
            mask_expanded = mask.unsqueeze(1)
            merged = merged.masked_fill(mask_expanded == 0, -1e9)
        
        # 全局最大池化: (batch, hidden_dim * 3, seq_len) -> (batch, hidden_dim * 3)
        pooled = torch.max(merged, dim=2)[0]
        
        # 得到最终 Logits
        logits = self.classifier(pooled)
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        return logits,probs, loss

class MeanFeatureModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes,dropout=0.4, eps=1e-5):
        super(MeanFeatureModule, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, num_classes)
        )

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.05,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=eps 
        )

    def forward(self, protein_features, labels):
        
        logits = self.classifier(protein_features)
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)

        return logits, probs, loss

class PrototypeModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, eps=1e-5):
        super(PrototypeModule, self).__init__()

        # 原型概率可学习的参数τ和bias
        self.base_tau = 5.0 # tau的缩放因子
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.zeros(num_classes))

        self.proto_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),           
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.05,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=eps 
        )

    def forward(self, protein_features, prototype_feats, labels):
        # 原型网络
        protein_features = F.normalize(self.proto_proj(protein_features), p=2, dim=1) # 先投影至同一空间，再归一化：消除模长干扰 
        prototype_feats = F.normalize(self.proto_proj(prototype_feats), p=2, dim=1)
        distances = torch.cdist(protein_features, prototype_feats, p=2.0) ** 2 # 计算欧氏距离的平方, distances[i, j] 表示第 i 个蛋白质到第 j 个原型的距离
        tau = -torch.exp(self.log_tau) * self.base_tau # tau 一定为负
        logits = tau * distances + self.b
        proto = torch.sigmoid(logits)
        loss = self.loss_fn(logits, labels)

        return proto, loss, tau

class Model(nn.Module):
    def __init__(self, prototype_feats, frequencies, rgcn_dim, labels_index, input_dim, hidden_dim, num_classes, eps=1e-4, node_feat_path="./new/data/go_embeddings.pt"):
        super(Model, self).__init__()

        rgcn_node_features = torch.load(node_feat_path, weights_only=True, map_location='cpu')
        self.register_buffer('go_features', rgcn_node_features[labels_index])
        self.register_buffer('prototype_feats', prototype_feats)
        self.LayerNorm = nn.LayerNorm(input_dim)
        self.residue_LayerNorm = nn.LayerNorm(input_dim)
        self.go_LayerNorm = nn.LayerNorm(rgcn_dim)

        self.frequencies = frequencies
        self.eps = eps # 用于稳定数值

        self.meanFeature_module = MeanFeatureModule(input_dim, hidden_dim, num_classes, eps=eps) # 平均特征模块
        self.maxFeature_module = MaxFeatureModule(input_dim, hidden_dim, num_classes, eps=eps) # 最大特征模块
        self.cross_attention_module = CrossAttentionModule(rgcn_dim, input_dim, hidden_dim, num_classes, eps=eps) # 跨模态注意力模块
        self.prototype_Module = PrototypeModule(input_dim, hidden_dim, num_classes, eps=eps) # 原型模块

        self.term_gate_weight = nn.Linear(rgcn_dim, 3)

        # 概率结合可学习的参数: 中心阈值c和平滑系数k
        self.gate_k = nn.Parameter(torch.tensor([2.0])) 
        self.gate_c_raw = nn.Parameter(torch.logit(self.frequencies.mean())) # 初始化 gate_c_raw，使得 gate_c 接近 frequencies 的平均值

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.05,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=eps 
        )

    def forward(self, protein_feats_data, residue_feats_data, mask, labels):

        protein_features = self.LayerNorm(protein_feats_data)
        prototype_feats = self.LayerNorm(self.prototype_feats)
        residue_feats_data = self.residue_LayerNorm(residue_feats_data)
        go_LayerNorm = self.go_LayerNorm(self.go_features)
        
        mean_logits, mean_probs, mean_loss = self.meanFeature_module(protein_features, labels)
        max_logits, max_probs, max_loss = self.maxFeature_module(residue_feats_data, mask, labels)
        cross_attn_logits, cross_attn_probs, cross_attn_loss = self.cross_attention_module(go_LayerNorm, residue_feats_data, mask, labels)
        proto_probs, proto_loss, tau = self.prototype_Module(protein_features, prototype_feats, labels)

        weights = torch.softmax(self.term_gate_weight(self.go_features), dim=-1)
        probs = weights[:, 0] * mean_probs + weights[:, 1] * max_probs + weights[:, 2] * cross_attn_probs

        # 联合概率
        gate_c = torch.sigmoid(self.gate_c_raw) # 范围在0-1之间
        sigma = 1.0 / (1.0 + torch.exp(-self.gate_k * (self.frequencies - gate_c)))
        final_probs = sigma * probs + (1.0 - sigma) * proto_probs

        # 通过概率倒推逻辑上的logits
        final_logits = torch.logit(final_probs, eps=self.eps)
        gate_loss = self.loss_fn(final_logits, labels)
        
        return mean_probs, max_probs, cross_attn_probs, proto_probs, final_probs, mean_loss, max_loss, cross_attn_loss, proto_loss, gate_loss, tau, self.prototype_Module.b, self.gate_k, gate_c, weights, sigma.mean()