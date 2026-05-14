import argparse
import torch
import utils
from models.RGCN import Model
from ruamel.yaml import YAML

def inference(model, graph, input_ids, edge_types):
    """
    推理函数：关闭梯度计算，返回节点的表征(embeddings)
    """
    model.eval()
    with torch.no_grad():
        # 根据原模型定义，前向传播返回 node_feats 和 loss
        # 这里我们只需要 node_feats
        node_feats, _ = model(graph, input_ids, edge_types)  
    
    return node_feats

def main(args, config):
    device = torch.device(args.device)
    utils.set_random_seed(args.seed) 

    num_nodes = config['num_nodes']
    h_dim = config['rgcn_dim']
    num_rels = config['num_rels']
    num_layers = config['num_layers']
    num_bases = config['num_bases']
    gamma = config['gamma']

    model = Model(num_nodes, h_dim, num_rels, num_layers, num_bases, gamma).to(device)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Checkpoint loaded! (Epoch: {checkpoint.get('epoch', 'N/A')})")

    datasets_path = args.path
    graph_file = datasets_path + "go_topology.pkl"
    
    graph, edge_types = utils.get_graph(graph_file, num_nodes)
    graph, edge_types = graph.to(device), edge_types.to(device)
    input_ids = torch.arange(num_nodes).to(device)

    if utils.is_main_process():      
        print('Starting inference......')
        
    node_embeddings = inference(model, graph, input_ids, edge_types)
    
    if utils.is_main_process():      
        print('Inference done!')
        print(f'Node embeddings shape: {node_embeddings.shape}')
        
        # 在这里保存 embeddings 或进行下游任务 (例如计算余弦相似度等)
        # mean_cos_sim, sim_variance = utils.compute_graph_metrics(node_embeddings, save_path = "/archive/hot3/fty/result/histogram/histogram_infer.png", sample_size=(len(input_ids)//10))
        torch.save(node_embeddings, './new/data/go_embeddings.pt')


if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='Inference Script')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./new/config/rgcn.yml', help='config/.yml')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--path', type=str, default="./new/data/", help='datasets path')
    parser.add_argument('--checkpoint', default='/archive/hot3/fty/rgcn_checkpoints/pubMedBert/checkpoint_13000.pth', type=str, help='path to the saved checkpoint (.pth file)')
    # parser.add_argument('--checkpoint', default='/archive/hot3/fty/rgcn_checkpoints/random/checkpoint_49999.pth', type=str, help='path to the saved checkpoint (.pth file)')

    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    
    main(args, config)