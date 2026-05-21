import torch
import os
import math
import random
import dgl
import numpy as np
import torch.distributed as dist
import pickle
import time
import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import plotly.express as px

from sklearn.manifold import TSNE
from adjustText import adjust_text
from statistics import mean
from scipy.special import expit
from collections import defaultdict, deque
from sklearn.metrics import precision_recall_curve, average_precision_score
from safetensors.torch import save_file, load_file

def lr_scheduler(current_epoch, warmup_epochs, max_epochs):
    """
    学习率调度逻辑：预热 + 余弦退火
    :param current_epoch: 当前的 epoch (由 LambdaLR 自动传入)
    :param warmup_epochs: 预热阶段的总 epoch 数
    :param max_epochs: 训练的总 epoch 数
    :return: 学习率乘法因子
    """
    # 线性预热阶段
    if current_epoch < warmup_epochs: 
        return float(current_epoch) / float(max(1, warmup_epochs))
    
    # 余弦退火阶段
    else: 
        progress = float(current_epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
        # 使用余弦公式计算缩放因子
        return 0.5 * (1.0 + math.cos(math.pi * progress))

def init_distributed_mode(args):

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def go2id(path):
    '''path为训练集中出现过的go term，直接根据顺序转为id'''

    with open(path, 'r') as f:
        go2id_dict = {line.strip(): i for i,line in enumerate(f)}

    return go2id_dict

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # 放开下面的注释会导致运行速度略微变慢，但会保证固定随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def is_main_process():
    """
    判断当前进程是否是主进程 (Global Rank 0)。
    
    返回:
        bool: 如果是主进程或非分布式环境返回 True，否则返回 False。
    """
    if not dist.is_available() or not dist.is_initialized():
        return True

    return dist.get_rank() == 0

def load_dict_from_pkl(file_path):

    with open(file_path, 'rb') as f:
        return pickle.load(f)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.6f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64).to('cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        if self.count == 0:
            return 0.0
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def max(self):
        if self.count == 0:
            return 0.0
        return max(self.deque)

    @property
    def value(self):
        if self.count == 0:
            return 0.0
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def calculate_metrics(y_true, input, is_logits=True):
    """
    y_true: ndarray, shape (n_samples, n_classes), multi-hot 编码
    logits: ndarray, shape (n_samples, n_classes), 模型原始输出
    """
    if is_logits:
        y_pred = expit(input)
    else:
        y_pred = input

    # --- Fmax 计算 ---
    # 计算所有可能的阈值下的 precision 和 recall
    # 展平处理以计算全局阈值 (CAFA 标准做法)
    precision, recall, thresholds = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    
    # 过滤掉分母为 0 的情况
    f1_scores = np.divide(2 * precision * recall, precision + recall, 
                          out=np.zeros_like(precision), where=(precision + recall) > 0)
    fmax = np.max(f1_scores)

    # --- AUPRC 计算 ---
    micro_auprc = average_precision_score(y_true, y_pred, average='micro')

    # macro AUPRC如果某个类别正样本数为0，则无法计算召回率(分母为0)，因此剔除正样本为0的类别。所以负样本的指标通过平均负样本得分来评判。
    mask = y_true.sum(axis=0) > 0
    y_true_mask = y_true[:, mask]
    y_pred_mask = y_pred[:, mask]
    macro_auprc = average_precision_score(y_true_mask, y_pred_mask, average='macro')

    # ---- 计算平均负样本得分 ----
    neg_scores = y_pred[y_true == 0]
    mean_neg_score = np.mean(neg_scores)

    return {
        "Fmax": fmax,
        "micro_AUPRC": micro_auprc,
        "macro_AUPRC": macro_auprc,
        "mean_neg_score": mean_neg_score
    }

def process_prototype(prototype_dict, go2id):
    '''将字典形式的原型处理为tensor，并与term_list中的顺序保持一致'''
    return torch.stack([prototype_dict[go] for go, id in go2id.items()])

def count_occurrences(file1_path, file2_path):
    '''
    根据term_list返回训练数据go term出现的频次
    file1_path: train_gene_label
    file2_path: term_list
    '''
    # 用来存储 S2 出现的次数
    # key: 字符串(S2), value: 出现的次数(count)
    s2_counter = {}
    res = {}

    # 1. 处理第一个文件
    with open(file1_path, 'r', encoding='utf-8') as f1:
        for line in f1:
            line = line.strip()
            if not line:
                continue
            
            # 以空格拆分，parts[0] 是 S1, parts[1] 是以逗号分隔的列表 L
            parts = line.split()
            if len(parts) < 2:
                continue
            
            # 获取列表 L 并按逗号拆分
            # 使用 set(l_list) 是为了防止同一个 S2 在同一个 S1 的列表中重复出现导致重复计数
            l_list = set(parts[1].split(','))
            
            # 统计每个元素出现的次数
            for item in l_list:
                s2_counter[item] = s2_counter.get(item, 0) + 1

    # 2. 处理第二个文件并打印结果
    with open(file2_path, 'r', encoding='utf-8') as f2:
        for line in f2:
            s2_query = line.strip()
            if not s2_query:
                continue
            
            # 从字典中获取统计结果，如果没出现过则为 0
            # print(s2_query, s2_counter.get(s2_query, 0))
            res[s2_query] = s2_counter.get(s2_query, 0)

    return res

def process_frequencies(freqs, go2id):
    '''将频率按照go2id的顺序处理成tensor的格式'''
    return torch.tensor([freqs[go] for go, id in go2id.items()])

def frequencies_projector(frequencies, eps = 1e-6):
    '''将数量级相差很大的频率数字投影至同一数量级并最大最小值归一化'''
    freq_log = torch.log(frequencies + 1) / torch.log(torch.tensor(10.0)) # 频率取log10(x + 1)，将高频低频缩放至同一数量级
    return (freq_log - freq_log.min()) / (freq_log.max() - freq_log.min() + eps) # 最大最小归一化

def draw_frequencies_AUPRC(final_probs, probs, y_true, frequencies, save_path, namespace=None):
    """
    绘制频率-APUR曲线图，横坐标频率，纵坐标对应频率的go term的平均AUPR
    """
    mask = y_true.sum(axis=0) > 0
    y_true_mask = y_true[:, mask]
    y_probs_mask = probs[:, mask]
    y_final_probs_mask = final_probs[:, mask]
    frequencies = frequencies[mask]
    # 1. 计算每个类别的 AUPR (average_precision_score 即为 AUPR)
    # 针对每一列（类别）计算
    aupr1 = [average_precision_score(y_true_mask[:, i], y_final_probs_mask[:, i]) for i in range(frequencies.shape[0])]
    aupr2 = [average_precision_score(y_true_mask[:, i], y_probs_mask[:, i]) for i in range(frequencies.shape[0])]
    
    # 2. 根据频率从小到大排序的索引
    sort_idx = np.argsort(frequencies)
    sorted_freq = frequencies[sort_idx]
    sorted_y1 = np.array(aupr1)[sort_idx]
    sorted_y2 = np.array(aupr2)[sort_idx]

    y1_dict, y2_dict = defaultdict(list), defaultdict(list)
    for i,x in enumerate(sorted_freq):
        y1_dict[x.item()].append(sorted_y1[i])
        y2_dict[x.item()].append(sorted_y2[i])

    y1 = [mean(y_list).item() for x, y_list in y1_dict.items()]
    y2 = [mean(y_list).item() for x, y_list in y2_dict.items()]
    x = [x for x, y_list in y2_dict.items()]

    # 3. 绘图
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线 1 并填充
    plt.plot(x, y1, label='our method', color='blue', lw=2)
    plt.fill_between(x, y1, color='blue', alpha=0.2)
    
    # 绘制曲线 2 并填充
    plt.plot(x, y2, label='mlp', color='red', lw=2)
    plt.fill_between(x, y2, color='red', alpha=0.2)

    # 装饰
    plt.xlabel('Class Frequency')
    plt.ylabel('AUPR')
    plt.title('AUPR vs Class Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(save_path, namespace + '_freq-AUPR.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

def get_graph(pkl_path, num_nodes):
    """
    从pkl文件加载数据并构建DGL图
    关系映射顺序：0: is_a, 1: part_of, 2: positively_regulates, 3: negatively_regulates,
                4: is_a_reverse, 5: part_of_reverse, 6: positively_regulates_reverse, 7: negatively_regulates_reverse
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 定义关系类型的固定顺序
    rel_order = [
        "is_a", "part_of", "positively_regulates", "negatively_regulates",
        "is_a_reverse", "part_of_reverse", "positively_regulates_reverse", "negatively_regulates_reverse"
    ]

    all_heads = []
    all_tails = []
    edge_types = []

    for i, rel_name in enumerate(rel_order):
        heads = data[rel_name]['head']
        tails = data[rel_name]['tail']
        
        all_heads.extend(heads)
        all_tails.extend(tails)
        # 生成对应的关系索引（0-7）
        edge_types.extend([i] * len(heads))

    # 构建DGL图
    g = dgl.graph((torch.tensor(all_heads), torch.tensor(all_tails)), num_nodes=num_nodes)
    
    # 将关系列表转换为张量
    e = torch.tensor(edge_types)

    return g, e

def load_id_mapping(file_path):

    result = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            if idx==1: continue
            line = line.strip()
            if not line:          # 跳过空行
                continue
            # 按逗号分割，最多分成两部分
            parts = line.split(',', 1)
            if len(parts) == 2:
                value, key = parts[0].strip(), parts[1].strip()
                result[key] = int(value)
    return result

def load_namespace(file_path):

    result = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            if idx==1: continue
            line = line.strip()
            if not line:          # 跳过空行
                continue
            # 按逗号分割，最多分成两部分
            parts = line.split(',', 1)
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip()
                result[key] = value
    return result

def split_graph(g, etypes, num_rels, train_ratio=0.8):
    """
    为RGCN链路预测构建训练集与测试集
    num_rels: 边类型的数量
    """

    # 获取正向边索引
    fwd_mask = (etypes >= 0) & (etypes < (num_rels/2))
    fwd_idx = torch.where(fwd_mask)[0]

    # 获取正向边的头节点、尾节点
    u, v = g.find_edges(fwd_idx)
    # 获取反向边索引
    bwd_idx = g.edge_ids(v, u) 
    
    # pairs每一行是 (正向边ID, 反向边ID)
    pairs = torch.stack([fwd_idx, bwd_idx], dim=1)

    # 随机打乱 Pair 级别的数据
    num_pairs = pairs.shape[0] # 根据pairs的维度一，即指的是头和尾(正向反向在维度二表示，维度一表示头和尾)
    indices = torch.randperm(num_pairs) # 随机
    train_size = int(num_pairs * train_ratio)

    train_pairs = pairs[indices[:train_size]]
    test_pairs = pairs[indices[train_size:]]

    # 将正向边反向边展平为一维 ID 列表，测试集索引和训练集索引，将pairs的两个维度展平为一维时，保证了同一对头尾节点的正向边和反向边不会泄露在训练集测试集两侧。
    train_ids = train_pairs.view(-1)
    test_ids = test_pairs.view(-1)

    # 返回测试集训练集
    u, v = g.edges()
    test_pos_u = u[test_ids]
    test_pos_v = v[test_ids]
    test_pos_etype = etypes[test_ids] 

    train_g = dgl.edge_subgraph(g, train_ids, relabel_nodes=False)
    train_g.edata['etype'] = etypes[train_ids]

    return train_g, (test_pos_u, test_pos_v, test_pos_etype)

def build_filter_dict(graph, edge_types):
    """
    构建用于 Filtered Rank 的真实三元组字典
    必须使用包含所有(Train+Test)边的全图来构建
    """
    filter_dict = defaultdict(set)
    u, v = graph.edges()
    
    # 将 tensor 转换为 list 加快构建速度
    u_list = u.tolist()
    v_list = v.tolist()
    r_list = edge_types.tolist()
    
    for head, tail, rel in zip(u_list, v_list, r_list):
        filter_dict[(head, rel)].add(tail)
        
    return filter_dict

def visualize_go_embeddings(embeddings, labels=None, names=None, save_path="go_tsne.png", title="t-SNE Visualization of GO Terms", plot_size = 15, fontsize=10, is_saveing_html = False):
    """
    使用 t-SNE 可视化 GO Term 嵌入
    :param embeddings: torch.Tensor 或 numpy.array, 形状为 [节点数, 嵌入维度]
    :param labels: 可选，list 或 array，用于给节点着色（例如 GO 的三个子类：BP, CC, MF）
    :param save_path: 图片保存路径
    :param title: 图表标题
    :param plot_size: 数据点大小
    """

    if isinstance(embeddings, torch.Tensor):
        data = embeddings.detach().cpu().numpy()
    else:
        data = embeddings

    print(f"正在进行 t-SNE 降维 (输入维度: {data.shape})...")

    # 配置 t-SNE， perplexity 建议在 30-50 之间；n_jobs=-1 使用所有 CPU 核心加速
    tsne = TSNE(
        n_components=2, 
        perplexity=30, 
        init='pca', 
        learning_rate='auto', 
        random_state=42
    )
    
    low_dim_embs = tsne.fit_transform(data)

    plt.figure(figsize=(12, 10), dpi=300)
    sns.set_style("whitegrid")
    
    colors = ["#E41A1C", "#0000FF", "#00CC00"] 
    if labels is not None:
        # 如果有标签，绘制带颜色的散点图
        sns.scatterplot(
            x=low_dim_embs[:, 0], 
            y=low_dim_embs[:, 1], 
            hue=labels, 
            palette=colors, 
            s=plot_size, 
            alpha=0.7, 
            edgecolor=None
        )
        plt.legend(title='Sub-Ontology', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # 如果没有标签，绘制单色散点图
        plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], s=10, alpha=0.6, c='#3498db')

    if names is not None and len(names) <= 200: # 设施plot数小于200时才会标注数据点的名称，否则太多了会导致图上很拥挤，看不出重点。
        texts = []
        for i, name in enumerate(names):
            # 这里的 i 对应的就是 low_dim_embs 的行索引
            texts.append(plt.text(low_dim_embs[i, 0], low_dim_embs[i, 1], name, fontsize=fontsize))
        
        # 自动调整标签位置，防止重叠
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    plt.title(title, fontsize=15)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    
    # 4. 保存并显示
    plt.tight_layout()
    plt.savefig(save_path + ".png")
    plt.show()

    if is_saveing_html:
        save_interactive_tsne_html(low_dim_embs, labels, names, save_path + ".html") # 可选方法：为T-SNE保存可交互的html，目的是观察聚类现象的点是哪些

def compute_graph_metrics(embeddings, save_path, sample_size=2000):
    """
    计算 Embedding 的健康度指标
    :param embeddings: Tensor (node_num, dim)
    :param sample_size: 采样节点数，用于计算两两相似度
    """
    node_num = embeddings.shape[0]
    indices = torch.randperm(node_num)[:sample_size]
    
    # 平均余弦相似度
    selected_emb = embeddings[indices]
    
    # L2 归一化后做内积即为余弦相似度
    norm_emb = F.normalize(selected_emb, p=2, dim=1)
    cos_sim_matrix = torch.mm(norm_emb, norm_emb.t())
    
    # 移除自相关的对角线元素，计算平均值
    mask = ~torch.eye(sample_size, dtype=torch.bool, device=embeddings.device)
    mean_cos_sim = cos_sim_matrix[mask].mean().item()

    # 计算方差
    sampled_data = embeddings[indices].flatten().cpu() # 展平为一维进行整体分布分析

    variance = torch.var(sampled_data).item()

    # 绘制相似度直方图
    sim_values_np = cos_sim_matrix[mask].cpu().numpy()

    plt.figure()
    plt.hist(sim_values_np, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title(f'Cosine Similarity Distribution\n(Mean: {mean_cos_sim:.4f})', fontsize=14)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.savefig(save_path + ".png")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.close() # 关闭图表防止内存溢出

    return mean_cos_sim, variance

def save_interactive_tsne_html(tsne_coords, labels, node_ids=None, output_filename="tsne_plot.html"):
    """
    将 T-SNE 降维后的坐标保存为可交互的 HTML 网页文件，便于在本地查看节点信息。
    
    参数:
    -----------
    tsne_coords : numpy.ndarray
        T-SNE 降维后的二维坐标矩阵，形状必须为 (node_num, 2)。
    
    labels : list 或 numpy.ndarray
        每个节点的类别标签（例如 ['BP', 'CC', 'MF', ...]），长度必须为 node_num。
        用于在图上区分颜色。
        
    node_ids : list 或 numpy.ndarray, 可选 (默认=None)
        你想要在鼠标悬停时显示的信息，长度必须为 node_num。
        - 如果不传（默认），鼠标悬停会显示该节点在矩阵中的行索引 (0, 1, 2...)。
        - 如果传入具体的 GO term 列表（如 ['GO:001', 'GO:002', ...]），悬停时会直接显示该 GO term。
        
    output_filename : str, 可选 (默认="tsne_plot.html")
        生成的 HTML 文件的保存路径和文件名。
    """
    
    node_num = tsne_coords.shape[0]
    
    # 如果没有提供 node_ids，则默认使用 0 到 node_num-1 的索引
    if node_ids is None:
        node_ids = np.arange(node_num)
        hover_name = "Index"
    else:
        hover_name = "GO_Term"
        
    # 1. 构建 DataFrame
    df = pd.DataFrame({
        'Dim_1': tsne_coords[:, 0],
        'Dim_2': tsne_coords[:, 1],
        'Ontology': labels,
        hover_name: node_ids
    })
    
    # 2. 绘制交互式散点图
    # hover_data 指定了鼠标悬停时要额外显示哪一列的数据
    fig = px.scatter(
        df, 
        x='Dim_1', 
        y='Dim_2', 
        color='Ontology', 
        hover_data=[hover_name],
        title="T-SNE Visualization of GO Terms (Interactive)"
    )
    
    # 3. 优化图片显示（让点稍微小一点、透明一点，在密集区域更容易看清）
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    
    # 4. 导出为 HTML
    fig.write_html(output_filename)

def get_labels_index(node_list_path, term_list_path):
    """从文件term_list中读取待预测的labels，返回对应于本体论层次图上的索引"""
    with open(term_list_path, 'r', encoding='utf-8') as f:
        term_list = [line.rstrip('\n') for line in f] 
    with open(node_list_path, 'r', encoding='utf-8') as f:
        node_list = [line.rstrip('\n').split(',')[1] for line in f][1:] 

    index_map = {value: idx for idx, value in enumerate(node_list)}
    return [index_map[item] for item in term_list]

def load_dict_from_safetensors(file_path):
    return load_file(file_path)

def get_mean_weights(w):
    """计算权重矩阵w的每列的平均值，返回一个一维张量，表示三个概率的权重平均值"""
    w = w.mean(dim=0).tolist()
    return ', '.join([f"{x:.4f}" for x in w])






