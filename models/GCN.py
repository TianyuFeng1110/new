import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

class GCNLayer(nn.Module):
    """带残差连接的预激活GCN层: Identity(x) + GCN(Dropout(GELU(LayerNorm(x))))"""

    def __init__(self, in_feat, out_feat, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_feat)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv = dglnn.GraphConv(in_feat, out_feat, norm='both', weight=True, bias=True)
        self.shortcut = nn.Linear(in_feat, out_feat) if in_feat != out_feat else nn.Identity()

    def forward(self, g, feat):
        residual = self.shortcut(feat)
        h = self.norm(feat)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv(g, h)
        return h + residual

class GoEmbeddingLayer(nn.Module):
    """GO术语嵌入层：多层GCN + 两种节点初始化方式"""

    def __init__(self, num_nodes, h_dim, num_layers=3, dropout=0.2,
                 edge_drop_prob=0.0, pretrained_path=None):
        super().__init__()

        if pretrained_path is not None:
            # 方式1：从外部文件加载预训练嵌入
            pretrained_feats = torch.load(pretrained_path, weights_only=True, map_location='cpu')
            self.node_emb = nn.Embedding.from_pretrained(pretrained_feats, freeze=False)
            self.proj = nn.Linear(pretrained_feats.shape[1], h_dim)
        else:
            # 方式2：随机初始化
            self.node_emb = nn.Embedding(num_nodes, h_dim)
            self.proj = nn.Identity()

        self.edge_drop_prob = edge_drop_prob
        self.layers = nn.ModuleList([
            GCNLayer(h_dim, h_dim, dropout) for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(h_dim)

    def forward(self, g, input_nodes_idx):
        if self.training and self.edge_drop_prob > 0.0:
            keep_idx = torch.rand(g.num_edges(), device=g.device) > self.edge_drop_prob
            g = dgl.edge_subgraph(g, keep_idx, relabel_nodes=False)

        h = self.node_emb(input_nodes_idx)
        h = self.proj(h)
        for layer in self.layers:
            h = layer(g, h)
        return self.out_norm(h)

class Model(nn.Module):
    """GCN嵌入模型 + 内积链接预测BCE损失"""

    def __init__(self, num_nodes, h_dim, num_layers=3, dropout=0.2,
                 edge_drop_prob=0.0, pretrained_path=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_embed = GoEmbeddingLayer(
            num_nodes, h_dim, num_layers, dropout, edge_drop_prob, pretrained_path
        )

    def forward(self, g, input_nodes_idx):
        node_feats = self.node_embed(g, input_nodes_idx)
        link_loss = self.compute_link_loss(g, node_feats)
        return node_feats, link_loss

    def compute_link_loss(self, g, node_feats, num_neg=1):
        """
        内积链接预测损失 (BCE with negative sampling)
        - 正样本: 图中真实存在的边
        - 负样本: 随机替换尾节点生成
        """
        u, v = g.edges()

        pos_score = (node_feats[u] * node_feats[v]).sum(dim=-1)
        neg_v = torch.randint(0, self.num_nodes, (u.shape[0] * num_neg,),
                              device=node_feats.device)
        neg_u = u.repeat_interleave(num_neg)
        neg_score = (node_feats[neg_u] * node_feats[neg_v]).sum(dim=-1)

        pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return pos_loss + neg_loss