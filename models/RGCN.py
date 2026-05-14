import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import random

# from models.xesm import ESM2
from dgl.nn import RelGraphConv
from transformers import EsmConfig
from torchvision.ops import sigmoid_focal_loss
from timm.loss import AsymmetricLossMultiLabel
from transformers import AutoTokenizer, AutoModel, EsmForMaskedLM, EsmModel

class PreActResRGCNLayer(nn.Module):
    """
    预激活残差RGCN层 (Pre-activation Residual RGCN Layer)
    结构: Identity(x) + RGCN(Dropout(GELU(LayerNorm(x))))
    """
    def __init__(self, in_feat, out_feat, num_rels, num_bases=None, dropout=0.2):
        super(PreActResRGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.norm = nn.LayerNorm(in_feat)

        self.act = nn.GELU()
        
        # RGCN，num_bases用于基分解正则化，防止关系过多导致参数爆炸
        self.conv = RelGraphConv(in_feat, out_feat, num_rels, 
                                 regularizer='basis' if num_bases else None,
                                 num_bases=num_bases,
                                 self_loop=True) # 自环有助于保留自身信息
        
        self.dropout = nn.Dropout(dropout)
        
        # (Residual connection)如果输入输出维度不一致，需要用线性层投影
        if in_feat != out_feat:
            self.shortcut = nn.Linear(in_feat, out_feat)
        else:
            self.shortcut = nn.Identity()

    def forward(self, g, feat, etypes):
        """
        g: DGLGraph
        feat: 节点特征矩阵
        etypes: 边类型张量
        """
        # 保存残差路径的输入
        residual = self.shortcut(feat)

        h = self.norm(feat)
        h = self.act(h)
        h = self.dropout(h) # Dropout常放在激活后或Conv后，这里放在Conv前也可以
        h = self.conv(g, h, etypes)
        
        # 残差连接
        return h + residual

class GoEmbedingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim, num_rels, num_layers=3, num_bases=4, edge_drop_prob=0.2, node_feat_path="./new/data/go_embeddings_init.pt"):
        super(GoEmbedingLayer, self).__init__()
        
        # self.node_emb = nn.Embedding(num_nodes, h_dim)
        initialized_node_feats = torch.load(node_feat_path, weights_only=True, map_location='cpu')
        self.node_emb = nn.Embedding.from_pretrained(initialized_node_feats, freeze=False)
        self.proj = nn.Linear(768, h_dim) # pubmedbert输出的特征维度为768
        
        self.layers = nn.ModuleList()

        # TODO 新增：整节点丢弃概率,多加的参数
        self.edge_drop_prob = edge_drop_prob 

        for i in range(num_layers):
            self.layers.append(
                PreActResRGCNLayer(h_dim, h_dim, num_rels, num_bases=num_bases)
            )
        self.norm = nn.LayerNorm(h_dim)

    def forward(self, g, input_nodes_idx, etypes):
        
        # --- TODO 新增 Edge Dropout 逻辑 ---
        if self.training and self.edge_drop_prob > 0.0:
            num_edges = g.num_edges()
            # 随机生成保留的边的索引
            keep_idx = torch.rand(num_edges, device=g.device) > self.edge_drop_prob
            
            # 过滤出保留的边和对应的边类型
            g = dgl.edge_subgraph(g, keep_idx, relabel_nodes=False)
            etypes = etypes[keep_idx]
        # ----------------------------

        # 获取节点的初始嵌入
        h = self.node_emb(input_nodes_idx) # 随机初始化 (node_num, h_dim) h_dim=320 
        h = self.proj(h) # 投影
        
        # 经过深层RGCN
        for layer in self.layers:
            h = layer(g, h, etypes)
            
        return self.norm(h)

class Model(nn.Module):
    def __init__(self, num_nodes, h_dim, num_rels, num_layers, num_bases, gamma=12.0):
        super(Model, self).__init__()
        
        # 确保 hidden dimension 可以被2整除（用于复数实部和虚部）
        assert h_dim % 2 == 0, "h_dim must be divisible by 2 for RotatE complex embedding."

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.gamma = gamma # RotatE的Margin超参数

        # =================GOA=====================
        # rotatE loss 关系的相位嵌入 
        self.rot_rel_emb = nn.Parameter(torch.zeros(num_rels, h_dim // 2))
        nn.init.uniform_(self.rot_rel_emb, -3.1415926, 3.1415926)

        self.node_embed = GoEmbedingLayer(num_nodes, h_dim, num_rels, num_layers, num_bases)

    def forward(self, g, input_nodes_idx, etypes): 
        
        node_feats = self.node_embed(g, input_nodes_idx, etypes) # 原始RGCN语义空间特征
        rotate_loss = self.compute_rotate_loss(g, node_feats, etypes) # rotatE loss，经过反向传递，此时的node_feats被优化至复数空间(从数学逻辑来看)
       
        return node_feats, rotate_loss

    def compute_rotate_loss(self, g, node_feats, etypes, num_neg=10):
        """
        计算 RotatE 损失
        g: DGLGraph, 包含当前的拓扑结构
        node_feats: RGCN 输出的节点嵌入 (将被视为复数)
        etypes: 边类型
        num_neg: 每个正样本对应的负样本数量
        """
        # 1. 获取图中的正样本三元组 (u, r, v)
        # 为了计算效率，可以只随机采样一部分边进行 Loss 计算，这里全量计算为例
        u, v = g.edges()
        r_idx = etypes # 边类型ID
        
        # 2. 准备数据
        batch_size = u.shape[0]
        # 拆分实部和虚部 [Batch, h_dim/2]
        # RGCN输出的 h_dim 维向量，前一半当做实部，后一半当做虚部
        re_head, im_head = torch.chunk(node_feats[u], 2, dim=-1)
        re_tail, im_tail = torch.chunk(node_feats[v], 2, dim=-1)
        
        # 获取关系的相位嵌入
        phase_rel = self.rot_rel_emb[r_idx] # [Batch, h_dim/2]
        
        # 3. 计算正样本分数 (Distance)
        # RotatE: h * r \approx t  => || h * r - t ||
        # Euler公式: r = cos(theta) + i*sin(theta)
        re_rel = torch.cos(phase_rel)
        im_rel = torch.sin(phase_rel)
        
        # 复数乘法: (a+bi)(c+di) = (ac-bd) + i(ad+bc)
        re_score = re_head * re_rel - im_head * im_rel
        im_score = re_head * im_rel + im_head * re_rel
        
        # 计算与尾节点的差值
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        
        # 取模 (L2 Norm) 得到距离
        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0) # [Batch, h_dim/2]
        score = score.sum(dim=-1) # [Batch]
        
        # 4. 负采样 (Negative Sampling)
        # 简单策略：随机替换尾节点 (t')
        # 生成 [Batch, Num_Neg] 个随机节点索引
        neg_idx = torch.randint(0, self.num_nodes, (batch_size, num_neg), device=node_feats.device)
        
        # 获取负样本的嵌入
        neg_emb = node_feats[neg_idx] # [Batch, Num_Neg, h_dim]
        re_neg_tail, im_neg_tail = torch.chunk(neg_emb, 2, dim=-1)
        
        # 广播正样本的预测结果 (h * r) 以便与多个负样本计算距离
        # re_pred: [Batch, 1, h_dim/2]
        re_pred = (re_head * re_rel - im_head * im_rel).unsqueeze(1)
        im_pred = (re_head * im_rel + im_head * re_rel).unsqueeze(1)
        
        re_score_neg = re_pred - re_neg_tail
        im_score_neg = im_pred - im_neg_tail
        
        score_neg = torch.stack([re_score_neg, im_score_neg], dim=0).norm(dim=0).sum(dim=-1) # [Batch, Num_Neg]
        
        # 5. 计算 Loss (Self-Adversarial Negative Sampling Loss)
        # Loss = -log sigmoid(gamma - pos_score) - sum( p(neg_score) * log sigmoid(neg_score - gamma) )
        
        # 正样本部分 (距离越小越好 -> gamma - distance 越大越好)
        pos_loss = -F.logsigmoid(self.gamma - score)
        
        # 负样本部分 (距离越大越好 -> distance - gamma 越大越好)
        # 计算自对抗权重
        neg_score_val = (self.gamma - score_neg) # 注意这里取反用于softmax
        neg_weights = F.softmax(neg_score_val, dim=1).detach()
        
        neg_loss = -torch.sum(neg_weights * F.logsigmoid(score_neg - self.gamma), dim=1)
        
        # 合并Loss
        loss = (pos_loss + neg_loss).mean()
        
        return loss