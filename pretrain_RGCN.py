import sys
import argparse
import torch
import torch.optim as optim
import ruamel.yaml as yaml
import new.utils as utils
import dgl
import os
import torch.nn.functional as F
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from new.models.GCN import Model
from ruamel.yaml import YAML

@torch.no_grad()
def evaluate(model, graph, input_ids, test_u, test_v, test_etype, filter_dict, batch_size, go2id, namesapce_dict, test_pairs, epoch):

    save_path = "/archive/hot3/fty/result"
    node_feats, loss = model(graph, input_ids, graph.edata['etype'])
    
    # 拆分全局实部和虚部 [N, h_dim/2]
    re_all, im_all = torch.chunk(node_feats, 2, dim=-1)
    
    ranks = []
    
    # 批量计算，防止内存/显存溢出 (OOM)
    for i in range(0, len(test_u), batch_size):
        u = test_u[i:i+batch_size]
        v = test_v[i:i+batch_size]
        r = test_etype[i:i+batch_size]

        re_head = re_all[u] 
        im_head = im_all[u] 
        phase_rel = model.rot_rel_emb[r] # 相位

        # 关系嵌入的实部re和虚部im
        re_rel = torch.cos(phase_rel)
        im_rel = torch.sin(phase_rel)

        # 预测的尾节点(h * r)，与真实尾结点计算差距 
        re_pred = re_head * re_rel - im_head * im_rel # [Batch, d]
        im_pred = re_head * im_rel + im_head * re_rel # [Batch, d]

        # 计算预测尾节点与全图中所有节点的 RotatE 距离 (利用广播机制)
        # re_pred: [Batch, 1, d], re_all: [1, N, d] -> 结果 [Batch, N, d]
        diff_re = re_pred.unsqueeze(1) - re_all.unsqueeze(0)
        diff_im = im_pred.unsqueeze(1) - im_all.unsqueeze(0)

        # 计算举例，距离越小越好 (L2 距离) -> [Batch, N]
        dist = torch.sqrt(diff_re**2 + diff_im**2 + 1e-8).sum(dim=-1)

        for b_idx in range(len(u)):
            head_idx = u[b_idx].item()
            rel_idx = r[b_idx].item()
            target_tail = v[b_idx].item()
            
            # 获取该 (head, rel) 下所有真实的 tail
            valid_tails = list(filter_dict.get((head_idx, rel_idx), set()))
            
            # 从中移除当前正在评估的目标 tail (因为我们要给它进行排名，不能过滤掉它)
            if target_tail in valid_tails:
                valid_tails.remove(target_tail)
            
            # 把其他真实存在的 tail 的距离设为正无穷大 (使其排在最后)
            if valid_tails:
                dist[b_idx, valid_tails] = float('inf')

        # 目标尾节点（正样本）的实际距离
        true_dist = dist[torch.arange(len(u)), v].unsqueeze(1)
        
        # 计算排名: 距离小于等于真实距离的节点个数
        rank = (dist < true_dist).sum(dim=1) + 1 
        ranks.append(rank)

    ranks = torch.cat(ranks).float()
    
    # 计算指标
    mrr = (1.0 / ranks).mean().item()
    hits_1 = (ranks <= 1).float().mean().item()
    hits_3 = (ranks <= 3).float().mean().item()
    hits_10 = (ranks <= 10).float().mean().item()
    
    # utils.calculate_sperman(test_pairs, node_feats.cpu(), go2id, save_path="./result_random/scatter_plot/similarity_scatter_plot" + str(epoch) + ".png") # 使用了rotate loss和正向反向边，resnik和余弦相似度就不再有意义，因为resnik是根据最近的祖先计算两个部分的拓扑重叠，而没有办法理解共同祖先通过正向调控与反向调控连接到两个节点上，他们其实表示的是相反的逻辑关系。

    # 对go集合的全集可视化
    utils.visualize_go_embeddings(node_feats, labels=[namespace for namespace in namesapce_dict.values()], names=list(go2id), save_path=save_path + "/t-SNE/all/t-SNE" + str(epoch), is_saveing_html=True)


    # 对go集合的子集goslim_agr可视化
    goslim_path = '/archive/hot3/fty/data/protein/goslim_agr.txt'
    with open(goslim_path, 'r', encoding='utf-8') as f:
        go_terms = [line.rstrip('\n').split(' ')[0] for line in f]
    slim_index = [go2id[go] for go in go_terms]
    namespace_list = [namespace for namespace in namesapce_dict.values()]
    namespace_slim_list = [namespace_list[i] for i in slim_index]
    utils.visualize_go_embeddings(node_feats[slim_index], labels=namespace_slim_list, names=go_terms, save_path= save_path + "/t-SNE/slim/t-SNE" + str(epoch), plot_size=150)
    
    # 随机采样计算平均余弦相似度，与其方差
    mean_cos_sim, sim_variance = utils.compute_graph_metrics(node_feats, save_path = save_path + "/histogram/histogram" + str(epoch), sample_size=(len(input_ids)//10))

    print(f"Eval | MRR: {mrr:.4f} | Hits@1: {hits_1:.4f} | Hits@3: {hits_3:.4f} | Hits@10: {hits_10:.4f} | Mean Cosine Similarity: {mean_cos_sim:.4f} | Cosine Similarity Variance: {sim_variance:.4f}")

def train(model, optimizer, graph, input_ids, edge_types, epoch):
    
    model.train()
    optimizer.zero_grad()
    node_feats, rotatE_loss = model(graph, input_ids, edge_types)  
    rotatE_loss.backward()
    print("Epoch {}, rotatE loss: {:.4f}".format(epoch, rotatE_loss))
    optimizer.step()  

def main(args, config):
    
    device = torch.device(args.device)
    seed = args.seed
    epochs = args.epochs
    datasets_path = args.path
    batch_size = args.batch_size
    graph_file = datasets_path + "go_topology.pkl"
    h_dim = config['rgcn_dim']
    num_rels = config['num_rels']
    num_layers = config['num_layers']
    num_bases = config['num_bases']
    gamma = config['gamma']
    graph_node_id_path = datasets_path + "graph_node_id.txt"
    graph_node_namespace_path = datasets_path + "graph_node_namespace.txt"
    go2id = utils.load_id_mapping(graph_node_id_path)
    namesapce_dict = utils.load_namespace(graph_node_namespace_path)
    num_nodes = len(go2id)

    # 设置随机种子
    utils.set_random_seed(seed) 

    # 加载模型
    model = Model(num_nodes, h_dim, num_rels, num_layers, num_bases, gamma).to(device)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    graph, edge_types = utils.get_graph(graph_file, num_nodes)
    graph, edge_types = graph.to(device), edge_types.to(device)
    input_ids = torch.arange(num_nodes).to(device)
    # 链路预测划分训练集和测试集
    train_g, test_set = utils.split_graph(graph, edge_types, num_rels)
    test_u, test_v, test_etype = [x.to(device) for x in test_set]
    filter_dict = utils.build_filter_dict(graph, edge_types)
    test_pairs=None
    if utils.is_main_process():      
        print('start traing......')
    for epoch in range(epochs):

        # 训练
        train(model, optimizer, train_g, input_ids, train_g.edata['etype'], epoch)
        # 评估
        if epoch%500==0 or epoch==epochs-1:
            evaluate(model, train_g, input_ids, test_u, test_v, test_etype, filter_dict, batch_size, go2id, namesapce_dict, test_pairs, epoch) # 链路预测，计算sperman
        
            if utils.is_main_process():                   
                save_obj = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                # torch.save(save_obj, os.path.join("./checkpoints/", 'checkpoint_%02d.pth'%epoch))  
                torch.save(save_obj, os.path.join("/archive/hot3/fty/rgcn_checkpoints/pubMedBert/", 'checkpoint_%02d.pth'%epoch))  

    if utils.is_main_process():      
        print('done!')
    
'''
使用PubMedBert作为go term节点的初始化特征，在基因本体层次图上使用RGCN学习拓扑关系。
'''
if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./new/config/rgcn.yml', help='config/.yml')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=50000, help='epoch')
    parser.add_argument('--path', type=str, default="./new/data/", help='datasets path')
    # parser.add_argument('--distributed', default=True, type=bool)
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    
    main(args, config)