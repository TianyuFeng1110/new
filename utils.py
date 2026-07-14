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
from collections import defaultdict, deque, Counter
from sklearn.metrics import precision_recall_curve, average_precision_score
from safetensors.torch import save_file, load_file
from goatools.obo_parser import GODag
from goatools.semantic import TermCounts
from typing import List, Dict
from sklearn.metrics import auc
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

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

def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def load_dict_from_pkl(file_path):

    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_data_from_pkl(file_path):

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

def get_ic(obo_path, term_list_path, gene_label_path):
    """
    计算基因本体术语的信息值 (Information Content, IC)，使用 goatools 库的 TermCounts 实现。
    IC 基于标注频率计算: IC(c) = -log( freq(c) / max_freq )，会自动沿 DAG 向上传播计数。

    :param obo_path: GO OBO 文件路径 (如 data/go-basic.obo)
    :param term_list_path: term_list 文件路径，每行一个 GO term
    :param gene_label_path: 训练集基因标注文件，每行格式: protein_id\\tGO:term1,GO:term2,...
    :return: numpy.ndarray, 形状 (n_terms,)，按 term_list 顺序排列的 IC 值
    NOTE 目前计算IC的方法是基于训练集的标注频率进行计算的，理论上应当使用gaf文件去计算
    """
    # 1. 加载 GO DAG
    godag = GODag(obo_path, optional_attrs=['relationship'])

    # 2. 读取 term_list
    with open(term_list_path, 'r') as f:
        term_list = [line.strip() for line in f if line.strip()]

    # 3. 构建 geneid -> set(GO IDs) 字典 (TermCounts 的 annots 参数要求 dict 格式)
    gene2gos = defaultdict(set)
    with open(gene_label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            protein_id = parts[0]
            go_terms = parts[1].split(',')
            gene2gos[protein_id].update(go_terms)

    # 4. 使用 TermCounts 沿 DAG 传播计数并计算 IC
    tcntobj = TermCounts(godag, gene2gos)

    # 5. 按 term_list 顺序计算 IC = -log(freq)，缺失项填 0.0
    ic_values = []
    for go_id in term_list:
        freq = tcntobj.get_term_freq(go_id)
        if freq > 0:
            ic_values.append(-math.log(freq))
        else:
            ic_values.append(0.0)

    return np.asarray(ic_values, dtype=np.float64)

# 以前的计算版本
def calculate_metrics_old(y_true, input, is_logits=False, ic=None):
    """
    y_true: ndarray, shape (n_samples, n_classes), multi-hot 编码
    input:  ndarray, shape (n_samples, n_classes), 模型原始输出
    ic:     ndarray, shape (n_classes,), 每个 GO term 的信息量 (Information Content).
            如果为 None，则根据 y_true 的标注频率自动计算: IC(c) = -log(freq(c)/max_freq).
    """
    if is_logits:
        y_pred = expit(input)
    else:
        y_pred = input

    n_samples, n_classes = y_true.shape

    # --- Fmax 计算 ---
    precision, recall, thresholds = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    f1_scores = np.divide(2 * precision * recall, precision + recall, 
                          out=np.zeros_like(precision), where=(precision + recall) > 0)
    fmax = np.max(f1_scores)

    # --- AUPRC 计算 ---
    micro_auprc = average_precision_score(y_true, y_pred, average='micro')

    mask = y_true.sum(axis=0) > 0
    y_true_mask = y_true[:, mask]
    y_pred_mask = y_pred[:, mask]
    macro_auprc = average_precision_score(y_true_mask, y_pred_mask, average='macro')

    # ---- 计算平均负样本得分 ----
    neg_scores = y_pred[y_true == 0]
    mean_neg_score = np.mean(neg_scores)

    if ic == None:
        return {
        "Fmax": fmax,
        "micro_AUPRC": micro_auprc,
        "macro_AUPRC": macro_auprc,
        "mean_neg_score": mean_neg_score,
    }

    # ---- WFmax & Smin (CAFA 标准, 101 个均匀阈值) ----
    n_t = 101
    thresholds = np.linspace(0.0, 1.0, n_t)

    ic_true = (y_true * ic).sum(axis=1)          # (n_samples,) 每蛋白真实标注总 IC
    total_ic = ic_true.sum()
    mask_true = ic_true > 0

    wf = np.zeros(n_t)
    ru = np.zeros(n_t)
    mi = np.zeros(n_t)

    for i, t in enumerate(thresholds):
        y_bin = (y_pred >= t).astype(np.float64)
        intersect = (y_bin * y_true * ic).sum(axis=1)
        pred_ic = (y_bin * ic).sum(axis=1)
        mask_pred = pred_ic > 0

        wPr = np.mean(intersect[mask_pred] / pred_ic[mask_pred]) if mask_pred.any() else 0.0
        wRc = np.mean(intersect[mask_true] / ic_true[mask_true]) if mask_true.any() else 0.0
        wf[i] = 2 * wPr * wRc / (wPr + wRc) if wPr + wRc > 0 else 0.0

        if total_ic > 0:
            ru[i] = (y_true * (1 - y_bin) * ic).sum() / total_ic
            mi[i] = ((1 - y_true) * y_bin * ic).sum() / total_ic

    wfmax = float(wf.max())
    smin = float(np.sqrt(ru ** 2 + mi ** 2).min())

    return {
        "Fmax": fmax,
        "micro_AUPRC": micro_auprc,
        "macro_AUPRC": macro_auprc,
        "mean_neg_score": mean_neg_score,
        "WFmax": wfmax,
        "Smin": smin
    }

def auprc(ytrue, ypred):
  p, r, t =  precision_recall_curve(ytrue, ypred)
  #print (r, len(r), p, t)
  return auc(r,p)

def macro_auprc(ytrue1, ypred1):
    """
    计算 Macro AUPRC（按类别平均的 AUPR，仅考虑在测试集中至少有一个正样本的类别）。

    :param ytrue1:    ndarray, shape (n_samples, n_classes), multi-hot 编码的真实标签
    :param ypred1:    ndarray, shape (n_samples, n_classes), 模型预测概率或 logits
    :return:          float, Macro AUPRC 值
    """

    # 只保留在测试集中至少有一个正样本的类别（全零列无法计算 AUPR）
    mask = ytrue1.sum(axis=0) > 0
    ytrue_mask = ytrue1[:, mask]
    ypred_mask = ypred1[:, mask]

    return average_precision_score(ytrue_mask, ypred_mask, average='macro')


def calculate_metrics(ytrue1, ypred1):

	fmax = 0
	prec_list = []
	recall_list=[]

	ytrue=[]
	ypred=[]
	# delete those sample whose labels are all 0.
	for i in range(len(ytrue1)):
		if np.sum(ytrue1[i]) >0:
			ytrue.append(ytrue1[i])
			ypred.append(ypred1[i])	

	# 将 list 转换为 ndarray（仅一次），避免循环内 100 次隐式转换
	ytrue = np.array(ytrue)
	ypred = np.array(ypred)

	for t in range(1, 101):
		thres = t/100.

		thres_array=np.ones((len(ytrue), len(ytrue[0])), dtype=np.float32) * thres

		pred_labels = np.greater(ypred, thres_array).astype(int)

		tp_matrix =pred_labels*ytrue

		tp = np.sum(tp_matrix, axis=1, dtype=np.int32)
		tpfp = np.sum(pred_labels, axis=1)
		tpfn = np.sum(ytrue,axis=1)

		avgprs=[]

		for i in range(len(tp)):
			if tpfp[i]!=0:
				avgprs.append(tp[i]/float(tpfp[i]))

		if len(avgprs)==0:
			continue
		avgpr = np.mean(avgprs)
		avgrc = np.mean(tp/tpfn)

		prec_list.append(avgpr)
		recall_list.append(avgrc)

		f1 = 2*avgpr*avgrc/(avgpr+avgrc)

		fmax=max(fmax, f1)

	return {"Fmax": fmax, "micro_AUPRC": auprc(ytrue.flatten(), ypred.flatten())}

# # 写的bug修复版本
# def auprc_new(ytrue, ypred):
#     """
#     使用官方标准的 average_precision_score (AP) 替代插值法 auc(r, p)
#     这能避免因梯形法则（线性插值）导致的 AUPRC 面积系统性高估问题。
#     """
#     return average_precision_score(ytrue, ypred)

# # 写的bug修复版本
# def calculate_metrics_new(ytrue1, ypred1):
#     fmax = 0.0
#     prec_list = []
#     recall_list = []

#     ytrue = []
#     ypred = []
#     # 1. 过滤掉那些全部为 0 的样本 (不包含真实标签的蛋白质不应参与评价)
#     for i in range(len(ytrue1)):
#         if np.sum(ytrue1[i]) > 0:
#             ytrue.append(ytrue1[i])
#             ypred.append(ypred1[i])    

#     # 【差异修复】将类型强转对齐为 float64
#     # 消除 Python 平台 float32 与浮点数阈值 t/100 比较时的底层精度误差问题
#     ytrue = np.array(ytrue, dtype=np.int32)
#     ypred = np.array(ypred, dtype=np.float64)
    
#     # 预先计算 TP + FN（代表该蛋白质具有的真实 GO Terms 数量）
#     # 这个值是不随阈值变化的，提前算好提高效率
#     tpfn = np.sum(ytrue, axis=1, dtype=np.float64)

#     for t in range(1, 101):
#         thres = t / 100.0

#         # 【差异修复】用 大于等于(>=) 替换严格大于(>)
#         # - 1e-7 是防浮点截断策略，保证 0.01 完全等同于 Matlab 计算下的 0.01
#         pred_labels = np.greater_equal(ypred, thres - 1e-7).astype(np.int32)

#         # 由于是向量化乘法，用 * 即可实现真阳性的筛选
#         tp_matrix = pred_labels * ytrue
        
#         tp = np.sum(tp_matrix, axis=1)         # 每个样本的真阳性(TP)数
#         tpfp = np.sum(pred_labels, axis=1)     # 每个样本预测出的正例总数(TP + FP)

#         # 按照 CAFA 的评估规则：
#         # Precision 仅评估在该阈值下 “至少做出了1个预测的蛋白质”
#         valid_indices = (tpfp != 0)
        
#         if not np.any(valid_indices):
#             # 如果当前阈值下，没有任何蛋白质有预测结果，跳过
#             continue
        
#         # 计算针对有效蛋白质的 平均 Precision
#         precisions = tp[valid_indices] / tpfp[valid_indices].astype(np.float64)
#         avgpr = np.mean(precisions)

#         # 按照 CAFA 的评估规则：
#         # Recall 即使某蛋白质预测数为0，它的 Recall(0) 也要被纳入均值计算
#         # 此前代码中的 np.mean(tp/tpfn) 就是对的，但用 np.float64 避免一些环境下的整除问题
#         recalls = tp / tpfn
#         avgrc = np.mean(recalls)

#         prec_list.append(avgpr)
#         recall_list.append(avgrc)

#         # 【差异修复】处理 Precision 和 Recall 同为 0 的极端情况
#         if avgpr + avgrc > 0:
#             f1 = 2 * avgpr * avgrc / (avgpr + avgrc)
#         else:
#             f1 = 0.0

#         # 不断刷新最大 F 得到 Fmax
#         fmax = max(fmax, f1)

#     return {"Fmax": fmax, "micro_AUPRC": auprc(ytrue.flatten(), ypred.flatten())}

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

# NOTE ：ds写的新的分片加载的方法
def new_load_dict_from_safetensors(path):
    """
    从 safetensors 文件中加载蛋白质特征字典。
    支持两种格式：
    - 单个 .safetensors 文件 (向后兼容旧格式)
    - 目录 (包含多个 part_XXXX.safetensors 分片文件)
    """
    if os.path.isdir(path):
        # 从分片目录加载：逐个读取 part_*.safetensors 并合并
        result = {}
        import glob
        part_files = sorted(glob.glob(os.path.join(path, 'part_*.safetensors')))
        if not part_files:
            raise FileNotFoundError(f'目录 {path} 中未找到 part_*.safetensors 分片文件')
        print(f'正在从 {len(part_files)} 个分片文件中加载特征...')
        for part_file in part_files:
            result.update(load_file(part_file))
        print(f'✅ 加载完成，共 {len(result)} 个蛋白质特征')
        return result
    else:
        return load_file(path)

def get_mean_weights(w):
    """计算权重矩阵w的每列的平均值，返回一个一维张量，表示三个概率的权重平均值"""
    w = w.mean(dim=(0, 1)).tolist()
    return ', '.join([f"{x:.4f}" for x in w])

# NOTE 此处是新数据的新方法
def build_term_list_from_cafa3(cafa3_path, namespace):
    """从 CAFA3 train_seq 中提取 GO term 列表（按 label 索引排序）。

    CAFA3 train_seq 中每条记录包含:
        'GO':    [GO ID 字符串, ...]
        'label': [整数索引, ...]    — 与该 GO 列表一一对应

    Returns:
        term_list: list[str]，按 label 索引排序的 GO ID 列表
        go2id:     dict[str -> int]，GO ID -> 类别索引
        idx2go:    dict[int -> str]，类别索引 -> GO ID（反向映射）
    """
    ns = namespace.lower()
    seq_path = os.path.join(cafa3_path, f"train_seq_{ns}")

    with open(seq_path, 'rb') as f:
        train_seqs = pickle.load(f)

    # go_index_pairs: list of (go_id, label_index)
    go_index_pairs = []
    for record in train_seqs:
        for go_id, label_idx in zip(record['GO'], record['label']):
            go_index_pairs.append((go_id, label_idx))

    # 按索引排序后去重
    go_index_pairs = sorted(set(go_index_pairs), key=lambda x: x[1])

    term_list = [go_id for go_id, _ in go_index_pairs]
    go2id = {go_id: idx for idx, go_id in enumerate(term_list)}
    idx2go = {idx: go_id for go_id, idx in go2id.items()}

    # 验证: term_list[i] 的 label 索引应该等于 i
    for i, (_, label_idx) in enumerate(go_index_pairs):
        assert label_idx == i, f"Label index mismatch at position {i}: {label_idx} != {i}"

    return term_list, go2id, idx2go

# NOTE 此处是新数据的新方法
def build_ic_from_cafa3(cafa3_path, namespace, idx2go, go2id, obo_path):
    """基于 CAFA3 训练标签计算 IC (Information Content)。

    使用 goatools 库，通过 obo 文件和标注计数计算每个 GO term 的 IC。

    Args:
        cafa3_path: CAFA3 数据集路径
        namespace:  BP/CC/MF
        idx2go:     dict[int -> str]，类别索引 -> GO ID
        go2id:      dict[str -> int]，GO ID -> 类别索引
        obo_path:   go-basic.obo 文件路径

    Returns:
        ic: ndarray[num_classes]
    """
    from goatools.obo_parser import GODag
    from goatools.semantic import TermCounts

    ns = namespace.lower()
    label_path = os.path.join(cafa3_path, f"train_label_{ns}")

    with open(label_path, 'rb') as f:
        labels = pickle.load(f)

    godag = GODag(obo_path, optional_attrs={'relationship'})

    # 统计每个 GO ID 的标注次数（O(1) 通过 idx2go 查询）
    go_counts = {}
    for label_indices in labels:
        for idx in label_indices:
            go_id = idx2go.get(idx)
            if go_id is not None:
                go_counts[go_id] = go_counts.get(go_id, 0) + 1

    term_counts = TermCounts(godag, go_counts)
    ic_dict = term_counts.get_information_content()

    ic = np.zeros(len(go2id))
    for go_id, cls_idx in go2id.items():
        ic[cls_idx] = ic_dict.get(go_id, 0.0)

    return ic

def compute_go_term_frequency(mask, num_proteins):
    '''获取每个go term的频率
       该go term注释到的蛋白质占训练集中总蛋白质的数量
    '''
    class_counts = mask.sum(dim=1).float()
    return class_counts / num_proteins, class_counts

def compute_protein_alf(protein_labels, go_frequencies):
    '''获取蛋白质的ALF'''
    
    if len(protein_labels) == 0:
        return 0.0  # 无标签的蛋白质，ALF定义为0
    return sum(go_frequencies[label] for label in protein_labels) / len(protein_labels)

def get_prototype_index(train_seq_cc):
    '''
    返回的list中每个元素表示一个go term，每一个元素是一个list，表示该go term注释到哪些蛋白质上
    索引与go term id保持一致
    '''

    # 获取GO term的最大索引（假定索引从0开始），以确定二维列表的行数
    max_go_idx = max((label for item in train_seq_cc for label in item.get('label', [])), default=-1)
    num_go_terms = max_go_idx + 1
    
    # 初始化二维列表，每一行对应一个go term，存放注释到该go term的蛋白质索引
    go_to_proteins = [[] for _ in range(num_go_terms)]
    
    # 建立映射：将蛋白质索引（即train_seq_cc的元素位置）存入对应的go term列表中
    for protein_idx, item in enumerate(train_seq_cc):
        for label in item.get('label', []):
            go_to_proteins[label].append(protein_idx)
            
    return go_to_proteins

def get_prototype_index_tensor(train_seq_cc):
    '''
    返回:
        padded_tensor: (num_go_terms, max_proteins) 的二维 LongTensor，
                       每一行对应该 GO term 注释到的蛋白质索引，不足位置用 -1 填充
        mask:          (num_go_terms, max_proteins) 的 BoolTensor，
                       True 表示有效蛋白质索引，False 表示填充位
    '''

    # 获取GO term的最大索引（假定索引从0开始），以确定二维列表的行数
    max_go_idx = max((label for item in train_seq_cc for label in item.get('label', [])), default=-1)
    num_go_terms = max_go_idx + 1

    # 初始化二维列表，每一行对应一个go term，存放注释到该go term的蛋白质索引
    go_to_proteins = [[] for _ in range(num_go_terms)]

    # 建立映射：将蛋白质索引（即train_seq_cc的元素位置）存入对应的go term列表中
    for protein_idx, item in enumerate(train_seq_cc):
        for label in item.get('label', []):
            go_to_proteins[label].append(protein_idx)

    # 计算最大蛋白质数，用于 padding
    max_proteins = max((len(prots) for prots in go_to_proteins), default=0)

    # 构建 padded tensor 和 mask
    padded_list = []
    mask_list = []
    for prots in go_to_proteins:
        n = len(prots)
        # 有效索引 + -1 填充
        row = prots + [-1] * (max_proteins - n)
        padded_list.append(row)
        # mask: 前 n 个为 True，其余为 False
        mask_row = [True] * n + [False] * (max_proteins - n)
        mask_list.append(mask_row)

    padded_tensor = torch.tensor(padded_list, dtype=torch.long)
    mask = torch.tensor(mask_list, dtype=torch.bool)

    return padded_tensor, mask

def compute_hier_loss(logits, parent_indices, child_indices):
    """R = 1/|E| * sum_{(i,j)∈E} max(0, y_j - y_i)，i父j子"""
    return torch.relu(logits[:, child_indices] - logits[:, parent_indices]).mean()


def balanced_bce_loss(logits, labels, mask, N, k):
    """BCE loss with padding removal + per-class averaging.

    Batch 结构: N 个类别 × 2k 个样本（k 正 + k 负）。
    策略 1 — 按类别求平均：每个类别内部先求 mean，再跨 N 个类别求 mean，
              确保罕见类与常见类对梯度的贡献为 1:1。
    策略 2 — 剔除 padding：mask=False 的占位样本不参与损失计算。

    Args:
        logits: [B, num_classes] 原始 logits
        labels: [B, num_classes] multi-hot 标签
        mask:   [B] bool, True=真实样本, False=padding 占位
        N:      每 batch 采样的类别数
        k:      每类正/负样本数上限
    Returns:
        scalar loss
    """
    device = logits.device
    samples_per_class = 2 * k  # k 正 + k 负

    # 从 batch 排列顺序重建类别归属: [0,0,...,1,1,...,N-1,N-1,...]
    class_ids = torch.arange(N, device=device).repeat_interleave(samples_per_class)

    # 策略 2: 剔除 padding 行
    class_ids = class_ids[mask]      # [B']
    logits = logits[mask]            # [B', num_classes]
    labels = labels[mask]            # [B', num_classes]

    # 逐元素 BCE (no reduction) → 每样本在所有类别维度上取 mean
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    per_sample_loss = bce.mean(dim=1)  # [B']

    # 策略 1: 按类别求平均
    loss = torch.zeros(N, device=device)
    for c in range(N):
        c_mask = (class_ids == c)
        if c_mask.sum() > 0:
            loss[c] = per_sample_loss[c_mask].mean()

    return loss.mean()


def balanced_asl_loss(logits, labels, mask, N, k,
                      gamma_neg=4, gamma_pos=0, clip=0.0, eps=1e-8):
    """ASL loss with padding removal + per-class averaging.

    与 balanced_bce_loss 逻辑一致，仅将 BCE 替换为 Asymmetric Loss。
    ASL 通过 gamma_neg 压制简单负样本的梯度，缓解正负极端不平衡问题。

    Args:
        logits:    [B, num_classes] 原始 logits
        labels:    [B, num_classes] multi-hot 标签
        mask:      [B] bool, True=真实样本, False=padding 占位
        N:         每 batch 采样的类别数
        k:         每类正/负样本数上限
        gamma_neg: 负样本 focusing 指数（默认 4，越大对简单负样本压制越强）
        gamma_pos: 正样本 focusing 指数（默认 0，标准 ASL 不对正样本加压）
        clip:      概率下界裁剪（默认 0.0，即不裁剪）
        eps:       数值稳定项
    Returns:
        scalar loss
    """
    device = logits.device
    samples_per_class = 2 * k

    class_ids = torch.arange(N, device=device).repeat_interleave(samples_per_class)

    # 策略 2: 剔除 padding 行
    class_ids = class_ids[mask]
    logits = logits[mask]
    labels = labels[mask]

    # 逐元素 ASL（手动实现，timm 的 AsymmetricLossMultiLabel 不支持 reduction='none'）
    p = torch.sigmoid(logits)
    # 正样本部分: -y * (1-p)^gamma_pos * log(p)
    los_pos = -labels * torch.pow(1 - p, gamma_pos) * torch.log(p.clamp(min=eps))
    # 负样本部分: -(1-y) * p_clipped^gamma_neg * log(1-p)
    p_neg = p
    if clip > 0:
        p_neg = (p_neg + clip).clamp(max=1)
    los_neg = -(1 - labels) * torch.pow(p_neg, gamma_neg) * torch.log((1 - p).clamp(min=eps))
    asl = los_pos + los_neg                                  # [B', num_classes]
    per_sample_loss = asl.mean(dim=1)                        # [B']

    # 策略 1: 按类别求平均
    loss = torch.zeros(N, device=device)
    for c in range(N):
        c_mask = (class_ids == c)
        if c_mask.sum() > 0:
            loss[c] = per_sample_loss[c_mask].mean()

    return loss.mean()

def get_hier_scale(model, train_loader, parent_indices, child_indices, device, num_calib_batches=10):
    """多 batch 静态标定：遍历多个 batch 累积梯度模长，取总模长比作为缩放因子，
    避免单 batch（尤其 batch_size=4~16）带来的高方差。
    """
    print('静态标定法获取梯度比例中...')
    model.train()  # 标定时需要 grad 可用
    params = [p for p in model.parameters() if p.requires_grad]

    custom_norms_sq = 0.0
    hier_norms_sq = 0.0
    actual_batches = 0

    for batch_idx, (batch_residue_feats, mask, labels, indices) in enumerate(train_loader):
        if batch_idx >= num_calib_batches:
            break

        # custom_logits, custom_loss = model(batch_residue_feats.to(device), mask.to(device), labels.to(device))
        final_probs, custom_logits, custom_loss, proto_loss, gate_loss, sigma = model(batch_residue_feats.to(device), indices.to(device), mask.to(device), labels.to(device))
        hier_loss = compute_hier_loss(custom_logits, parent_indices, child_indices)

        grad_custom = torch.autograd.grad(custom_loss, params, retain_graph=True, allow_unused=True)
        grad_hier = torch.autograd.grad(hier_loss, params, retain_graph=False, allow_unused=True)

        c_norm = sum(g.norm().item() ** 2 for g in grad_custom if g is not None)
        h_norm = sum(g.norm().item() ** 2 for g in grad_hier if g is not None)

        # 跳过梯度为 0 的退化 batch
        if h_norm < 1e-20:
            continue

        custom_norms_sq += c_norm
        hier_norms_sq += h_norm
        actual_batches += 1

    model.zero_grad(set_to_none=True)

    if hier_norms_sq < 1e-12:
        print(f"[Calibrate] total hier_grad_norm too small ({hier_norms_sq ** 0.5:.2e}), using scale=1.0")
        return 1.0

    scale = (custom_norms_sq / hier_norms_sq) ** 0.5
    print(f"[Calibrate] {actual_batches} batches, "
          f"total_custom_grad_norm={custom_norms_sq ** 0.5:.4f}, "
          f"total_hier_grad_norm={hier_norms_sq ** 0.5:.4f}, "
          f"hier_scale={scale:.4f}")
    return scale


def print_loss_gradients(model, train_loader, parent_indices, child_indices, device, num_batches=5):
    """打印各 loss 对全部参数的梯度 L2 范数（不做任何标定建议，仅输出原始数据）。

    输出每 batch 的梯度范数以及平均值，方便手动调整 scale。

    注意：由于 forward 中 custom_probs/proto_probs 有 .detach()，
    gate_loss 梯度只流向 gate_k、gate_c，其范数会远小于其他 loss。
    """
    print('=' * 60)
    print('各 Loss 梯度 L2 范数（原始值，未加权）')
    print('=' * 60)

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    records = []
    for batch_idx, (batch_residue_feats, labels, indices) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        final_probs, custom_logits, custom_loss, proto_loss, gate_loss, sigma, _, _ = model(
            batch_residue_feats.to(device), indices.to(device), labels.to(device)
        )
        hier_loss = compute_hier_loss(custom_logits, parent_indices, child_indices)

        grad_custom = torch.autograd.grad(custom_loss, params, retain_graph=True, allow_unused=True)
        grad_proto  = torch.autograd.grad(proto_loss,  params, retain_graph=True, allow_unused=True)
        grad_gate   = torch.autograd.grad(gate_loss,   params, retain_graph=True, allow_unused=True)
        grad_hier   = torch.autograd.grad(hier_loss,   params, retain_graph=False, allow_unused=True)

        c_norm = sum(g.data.norm().item() ** 2 for g in grad_custom if g is not None) ** 0.5
        p_norm = sum(g.data.norm().item() ** 2 for g in grad_proto  if g is not None) ** 0.5
        g_norm = sum(g.data.norm().item() ** 2 for g in grad_gate   if g is not None) ** 0.5
        h_norm = sum(g.data.norm().item() ** 2 for g in grad_hier   if g is not None) ** 0.5

        records.append((c_norm, p_norm, g_norm, h_norm))
        print(f'  batch {batch_idx:2d}: '
              f'custom_grad={c_norm:.4f}  '
              f'proto_grad={p_norm:.4f}  '
              f'gate_grad={g_norm:.4f}  '
              f'hier_grad={h_norm:.4f}')

    model.zero_grad(set_to_none=True)

    if records:
        avg_c = sum(r[0] for r in records) / len(records)
        avg_p = sum(r[1] for r in records) / len(records)
        avg_g = sum(r[2] for r in records) / len(records)
        avg_h = sum(r[3] for r in records) / len(records)
        print('-' * 60)
        print(f'  平均值:    '
              f'custom_grad={avg_c:.4f}  '
              f'proto_grad={avg_p:.4f}  '
              f'gate_grad={avg_g:.4f}  '
              f'hier_grad={avg_h:.4f}')
    print('=' * 60)


def eval_func_generalizability(model, test_loader, device, go_freq, prototypes=None):
    print("Starting evaluation...")

    # 1. 收集所有预测概率和真实标签
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_feats, labels, indices in test_loader:
            batch_feats = batch_feats.to(device)
            labels = labels.to(device)
            indices = indices.to(device)

            if prototypes is not None:
                # res = model(batch_feats, labels, None, None, prototypes)
                res = model(batch_feats, prototypes, labels)
            else:
                # res = model(batch_feats, labels)
                res = model(batch_feats, indices, labels)
            final_probs = res[0]

            all_probs.append(final_probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()   # (N, num_classes)
    all_labels = torch.cat(all_labels, dim=0).numpy()  # (N, num_classes)

    # 计算每个蛋白质的 ALF 
    go_freq_np = go_freq.cpu().numpy()  # (num_classes,)

    alfs = []
    for i in range(all_labels.shape[0]):
        label_indices = np.where(all_labels[i] > 0)[0]
        alf = compute_protein_alf(label_indices.tolist(), go_freq_np)
        alfs.append(alf)
    alfs = np.array(alfs)

    # 按 ALF 区间划分，计算各区间的 Fmax
    #    区间定义: [0, 0.2], (0.2, 0.3], (0.3, 0.4], (0.4, 1]
    #    为避免边界重叠，第一个区间闭区间，后续区间左开右闭
    bins = [
        (0.0, 0.2, "[0, 0.2]"),
        (0.2, 0.3, "(0.2, 0.3]"),
        (0.3, 0.4, "(0.3, 0.4]"),
        (0.4, 1.0, "(0.4, 1]"),
    ]

    print("\n===== ALF-based Fmax Evaluation =====")
    for low, high, bin_label in bins:
        if low == 0.0:
            mask = (alfs >= low) & (alfs <= high)
        else:
            mask = (alfs > low) & (alfs <= high)

        n_proteins = mask.sum()
        if n_proteins == 0:
            print(f"  ALF {bin_label}: No proteins in this bin")
            continue

        bin_probs = all_probs[mask]
        bin_labels = all_labels[mask]
        metrics = calculate_metrics(bin_labels, bin_probs)
        print(f"  ALF {bin_label}: n={n_proteins}, "
              f"Fmax={metrics['Fmax']:.4f}, micro_AUPRC={metrics['micro_AUPRC']:.4f}")

    metrics_all = calculate_metrics(all_labels, all_probs)
    print(f"\n  Overall: Fmax={metrics_all['Fmax']:.4f}, "
          f"micro_AUPRC={metrics_all['micro_AUPRC']:.4f}")


def eval_term_freq_generalizability(model, test_loader, device, go_freq, prototypes=None):
    """按 GO term 自身的功能频率划分频率桶，评估模型在不同频率桶上的泛化能力。

    与 eval_func_generalizability 的区别：
    - eval_func_generalizability 按蛋白质的 ALF（平均标签频率）给蛋白质分桶（行切片），
      计算每个桶的 Fmax / micro_AUPRC
    - 本方法按每个 GO term 在训练集中的频率 f(x) 给 GO term 分桶（列切片），
      同样计算每个桶的 Fmax / micro_AUPRC，同时输出 mean per-class AUPRC

    Args:
        model:       待评估模型
        test_loader: 测试集 DataLoader，每个 batch 返回 (batch_feats, labels, indices)
        device:      torch device
        go_freq:     Tensor (num_classes,)，每个 GO term 的训练集频率 f(x)
    """
    print("Starting GO-term frequency-based evaluation...")

    # 1. 收集所有预测概率和真实标签
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_feats, labels, indices in test_loader:
            batch_feats = batch_feats.to(device)
            labels = labels.to(device)
            indices = indices.to(device)

            if prototypes != None:
                # res = model(query=batch_feats, labels=labels, prototypes=prototypes)
                res = model(input_feats=batch_feats, labels=labels, prototypes=prototypes)
            else:
                res = model(batch_feats, indices, labels)
            final_probs = res[0]

            all_probs.append(final_probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()   # (N, num_classes)
    all_labels = torch.cat(all_labels, dim=0).numpy()  # (N, num_classes)

    go_freq_np = go_freq.cpu().numpy()  # (num_classes,)

    # 2. 按 GO term 频率划分区间
    # 0-10,10-50,50-100，100-1000，1000-all
    bins = [
        (0.0, 0.0002, "[0, 0.0002]"),
        (0.0002, 0.001, "(0.0002, 0.001]"),
        (0.001, 0.002, "(0.001, 0.002]"),
        (0.002, 0.02, "(0.002, 0.02]"),
        (0.02, 1.0, "(0.02, 1]"),
    ]

    print("\n===== GO-term Frequency-based Evaluation =====")
    for low, high, bin_label in bins:
        if low == 0.0:
            col_mask = (go_freq_np >= low) & (go_freq_np <= high)
        else:
            col_mask = (go_freq_np > low) & (go_freq_np <= high)

        col_indices = np.where(col_mask)[0]
        n_terms = len(col_indices)
        if n_terms == 0:
            print(f"  Freq {bin_label}: No GO terms in this bin")
            continue

        # 列切片：只保留该频率桶内的 GO term
        bin_probs = all_probs[:, col_indices]
        bin_labels = all_labels[:, col_indices]
        metrics = calculate_metrics(bin_labels, bin_probs)
        print(f"  Freq {bin_label}: n_terms={n_terms}, "
              f"Fmax={metrics['Fmax']:.4f}, micro_AUPRC={metrics['micro_AUPRC']:.4f}")

    # 3. 全量（所有 GO term）的指标
    metrics_all = calculate_metrics(all_labels, all_probs)
    print(f"\n  Overall: Fmax={metrics_all['Fmax']:.4f}, "
          f"micro_AUPRC={metrics_all['micro_AUPRC']:.4f}")

def get_prototype_index(data, num_classes):
    prototype_index = torch.zeros(len(data), num_classes)
    for i, item in enumerate(data):
        labels = item.get('label', [])
        # 使用高级索引将对应位置置为 1
        prototype_index[i, labels] = 1.0

    return prototype_index

def get_hard_neg_indices(pos_indices, go2id):
    """
    基于基因本体 DAG 层次结构获取困难负样本索引（仅直接父节点）。

    对于 GO term B，其困难负样本定义为：
      注释到 B 的 **直接父节点** (father) 的蛋白质，
      但 **未** 注释到 B 本身。

    直觉：蛋白质具有父节点的泛化功能，但不具有当前子节点的特化功能，
    这类"有泛化但缺特化"的样本对分类器最具迷惑性（困难负样本）。

    示例：
      C (父) → B → A (子)，即 C 是 B 的直接父节点，B 是 A 的直接父节点。
      对于 GO term B，困难负样本 = 注释到 C (直接父节点) 的蛋白质，但未注释到 B。

    Args:
        pos_indices: Tensor, shape (num_classes, num_data), 二值矩阵，
                     pos_indices[i][j] = 1 表示蛋白质 j 是第 i 个类的正样本。
        go2id:       dict, key 为 GO term 字符串 (如 'GO:0000109')，
                     value 为 dict { 'father': [str], 'child': [str], 'ind': int }。
                     'ind' 是该 GO term 在 num_classes 中的索引。

    Returns:
        hard_neg_indices: Tensor, shape (num_classes, num_data), 二值矩阵，
                          hard_neg_indices[i][j] = 1 表示蛋白质 j 是第 i 个类的困难负样本。
    """
    num_classes, num_data = pos_indices.shape

    # 1. 构建 class_index -> GO term string 的反向映射
    idx2go = {} # {类别索引: GO term}
    for go_str, info in go2id.items():
        if isinstance(info, dict) and 'ind' in info:
            idx2go[info['ind']] = go_str

    # 2. 构建困难负样本矩阵：仅使用直接父节点
    hard_neg_indices = torch.zeros(num_classes, num_data, dtype=pos_indices.dtype)

    for i in range(num_classes):
        go_str = idx2go.get(i)

        fathers = go2id[go_str].get('father', [])
        # 收集直接父节点的类别索引
        father_indices = []
        for f in fathers:
            if f in go2id:
                f_info = go2id[f]
                if isinstance(f_info, dict) and 'ind' in f_info:
                    father_indices.append(f_info['ind'])

        if not father_indices:
            continue

        # 表示当前类别i父节点的所有正样本
        father_pos = pos_indices[father_indices].any(dim=0)  # (num_data,)
        # 当前类自身为负 → 未注释到该类
        class_neg = (pos_indices[i] == 0)  # (num_data,)
        hard_neg_indices[i] = (father_pos & class_neg).to(pos_indices.dtype)

    return hard_neg_indices

def get_queue_index(features, one_hot, namespace, datasets_path, K=16):
    '''
    为每个类别静态选取入队的索引(该队列用于前向传递后计算原型)
    '''
    # 如果已经保存了队列索引，直接加载，否则实时计算，然后保存。
    file_path = os.path.join(datasets_path, f'{namespace.lower()}_queue_index.pt')
    if os.path.exists(file_path):
        return torch.load(file_path)

    classes_num = one_hot.size(1)
    queue_index = []

    for c in range(classes_num):
        # 1. 获取当前类别在原始 features 中的全局索引 (1D Tensor)
        # 例如，如果原始第 2, 5, 8 个样本属于该类别，则 global_indices = tensor([2, 5, 8])
        global_indices = (one_hot[:, c] == 1).nonzero(as_tuple=True)[0]
        
        # 2. 提取对应的特征向量
        f = features[global_indices]
        
        f_norm = F.normalize(f, p=2, dim=1)
        
        if f_norm.size(0) >= K:
            labels, centroids, closest_indices = kmeans_sklearn(f_norm, K)
            # closest_indices 通常是 list 或 numpy 数组，表示在当前类别子集中的索引
            # 3. 映射回全局索引：利用 PyTorch 的花式索引（Fancy Indexing）
            global_mapped = global_indices[closest_indices]
        else:
            # 当数量少于 K 时，保留该类别所有的样本，其对应的全局索引即为 global_indices
            # global_mapped = global_indices
            pad_len = max(0, K - global_indices.size(0))
            global_mapped = F.pad(global_indices, (0, pad_len), mode='constant', value=-1)
            
        # 根据您的需要，可以将映射后的全局索引保存
        queue_index.append(global_mapped)  # 按类别保存为 list of list
        # 或者使用 queue_index.extend(global_mapped) 合并为一个一维列表
        
    res = torch.stack(queue_index, dim=0)
    torch.save(res, file_path)

    return res

def kmeans_sklearn(X: torch.Tensor, K: int = 16):
    """对输入的张量进行 K-means 聚类，并寻找距离每个聚类形心最近的数据索引。

    参数:
        X: 形状为 (data_num, dim) 的数据，可以是 PyTorch 张量或 NumPy 数组。
        n_clusters: 聚类类别数，默认为 16。

    返回:
        labels: 每个数据点对应的聚类类别标签，形状为 (data_num,)。
        centers: 聚类形心，形状为 (n_clusters, dim)。
        closest_indices: 每个聚类类别中，距离形心最近的数据索引，形状为 (n_clusters,)。
    """
    # 如果输入是 PyTorch 张量，转换为 NumPy 数组以便 scikit-learn 处理
    if hasattr(X, "detach"):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)

    # 1. 实例化并进行 K-means 聚类
    kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_np)
    centers = kmeans.cluster_centers_

    # 2. 寻找距离每个聚类形心最近的数据点索引
    # pairwise_distances_argmin_min(A, B) 会计算 A 中每个元素到 B 中所有元素的最短距离
    # 这里 A 为形心(centers)，B 为原始数据集(X_np)，返回的第一个值即为最接近形心的数据点索引
    closest_indices, _ = pairwise_distances_argmin_min(centers, X_np)

    return labels, centers, closest_indices


@torch.no_grad()
def analyze_queue_coverage(model, go_freq):
    """统计每个类的队列中实际正样本数"""
    idx = model.queue_indices  # (num_classes, queue_size)
    valid_counts = (idx >= 0).sum(dim=1).cpu().numpy()  # 每个类队列中的有效样本数
    
    # 按频率分桶统计
    for (bucket_name, lo, hi) in [("rare [0,0.01)", 0, 0.01), 
                                   ("mid [0.01,0.1)", 0.01, 0.1),
                                   ("freq [0.1,1]", 0.1, 1.0)]:
        mask = (go_freq >= lo) & (go_freq < hi)
        if mask.sum() > 0:
            print(f"{bucket_name}: mean_valid={valid_counts[mask].mean():.1f}, "
                  f"zero_count={(valid_counts[mask] == 0).sum()}/{mask.sum()}")

@torch.no_grad()
def analyze_proto_quality(model, loader, device, go_freq, num_classes):
    """对每个类分别统计：正样本与原型、负样本与原型的余弦相似度分布"""
    model.eval()
    pos_cos = [[] for _ in range(num_classes)]
    neg_cos = [[] for _ in range(num_classes)]
    
    for protein_feats, labels, indices in loader:
        protein_feats = model.protein_mlp(protein_feats.to(device))  # (B, D)
        go_feats = model.go_feats                                    # (C, D)
        func_specific = protein_feats.unsqueeze(1) * go_feats.unsqueeze(0)  # (B, C, D)
        prototypes = model._compute_prototypes()                     # (C, D)
        
        func_norm = F.normalize(func_specific, p=2, dim=-1)
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        cos_sim = (func_norm * proto_norm.unsqueeze(0)).sum(dim=-1)  # (B, C)
        
        for c in range(num_classes):
            mask_pos = labels[:, c] == 1
            mask_neg = labels[:, c] == 0
            if mask_pos.sum() > 0:
                pos_cos[c].extend(cos_sim[mask_pos, c].cpu().tolist())
            if mask_neg.sum() > 0:
                neg_cos[c].extend(cos_sim[mask_neg, c].cpu().tolist())
    
    # 按频率分桶汇总
    for (bucket_name, lo, hi) in [("rare", 0, 0.01), ("mid", 0.01, 0.1), ("freq", 0.1, 1.0)]:
        mask = (go_freq >= lo) & (go_freq < hi)
        classes_in_bucket = mask.nonzero(as_tuple=True)[0]
        pos_all = np.concatenate([np.array(pos_cos[c]) for c in classes_in_bucket if len(pos_cos[c]) > 0])
        neg_all = np.concatenate([np.array(neg_cos[c]) for c in classes_in_bucket if len(neg_cos[c]) > 0])
        if len(pos_all) > 0 and len(neg_all) > 0:
            print(f"{bucket_name}: pos_cos_mean={pos_all.mean():.4f}, neg_cos_mean={neg_all.mean():.4f}, "
                  f"separation={pos_all.mean() - neg_all.mean():.4f}")
            
@torch.no_grad()
def analyze_proto_collapse(model, go_freq):
    """计算原型之间的两两余弦相似度，检查是否坍缩"""
    prototypes = F.normalize(model._compute_prototypes(), p=2, dim=-1)  # (C, D)
    sim_matrix = prototypes @ prototypes.T  # (C, C)
    
    for (bucket_name, lo, hi) in [("rare", 0, 0.01), ("mid", 0.01, 0.1), ("freq", 0.1, 1.0)]:
        mask = (go_freq >= lo) & (go_freq < hi)
        classes = mask.nonzero(as_tuple=True)[0]
        if len(classes) > 1:
            intra = sim_matrix[classes][:, classes]  # 同类桶内
            # 桶内类间平均相似度
            n = len(classes)
            intra_sim = (intra.sum() - intra.diag().sum()) / (n * (n - 1))
            
            # 与高频类的相似度
            freq_mask = go_freq >= 0.1
            freq_classes = freq_mask.nonzero(as_tuple=True)[0]
            if len(freq_classes) > 0:
                cross_sim = sim_matrix[classes][:, freq_classes].mean()
            else:
                cross_sim = 0
            print(f"{bucket_name} (n={n}): intra_sim={intra_sim:.4f}, cross_with_freq={cross_sim:.4f}")

@torch.no_grad()
def analyze_go_separation(model, go_freq):
    go_feats = model.go_feats  # (C, D)
    go_norm = F.normalize(go_feats, dim=-1)
    sim = go_norm @ go_norm.T  # (C, C)
    
    for (name, lo, hi) in [("rare", 0, 0.01), ("mid", 0.01, 0.1), ("freq", 0.1, 1)]:
        mask = (go_freq >= lo) & (go_freq < hi)
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) > 1:
            intra = (sim[idx][:, idx].sum() - len(idx)) / (len(idx) * (len(idx) - 1))
            print(f"{name} (n={len(idx)}): intra_go_sim={intra:.4f}")

    # rare vs rare, freq vs freq, rare vs freq
    rare_mask = go_freq < 0.01
    freq_mask = go_freq >= 0.1
    print(f"rare-freq cross go_sim: {sim[rare_mask][:, freq_mask].mean():.4f}")

@torch.no_grad()
def diagnose_prototype_failure(model, data_loader, device, go_freq, thresholds=(0.01, 0.05)):
    """
    诊断原型模块失效的核心原因：
    1. 验证哈达玛积后的特征范数 (是否因GO特征坍缩导致乘积后特征消失)
    2. 验证主网络与动量网络脱节程度 (是否发生严重的原型漂移)
    
    参数:
    thresholds: 用于划分 rare, mid, freq 的频率阈值。请根据你数据集中 n=2450, 100, 24 的真实分界线微调。
    """
    model.eval()
    
    # 获取单个 Batch 用于验证猜想1
    protein_feats, _, _ = next(iter(data_loader))
    protein_feats = protein_feats.to(device)
    
    # ---------------------------------------------------------
    # 猜想 1: 观察哈达玛积后、L2归一化前的特征范数 (L2 Norm)
    # ---------------------------------------------------------
    p_feats = model.protein_mlp(protein_feats)
    g_feats = model.go_feats
    
    # 哈达玛积得到的原始特异性特征 (未归一化)
    func_specific_raw = p_feats.unsqueeze(1) * g_feats.unsqueeze(0)
    
    # 计算每个 GO term (dim=1) 在当前 Batch (dim=0) 上的平均 L2 范数
    norms = torch.norm(func_specific_raw, p=2, dim=-1).mean(dim=0)
    norms_np = norms.cpu().numpy()
    
    # ---------------------------------------------------------
    # 猜想 2: 主网络 vs 动量网络在 GO 特征映射上的脱节程度
    # ---------------------------------------------------------
    main_go = model.go_feats
    mom_go = model.momentum_go_feats
    
    # 计算对应类别的余弦相似度
    drift_sim = F.cosine_similarity(main_go, mom_go, dim=-1)
    drift_sim_np = drift_sim.cpu().numpy()
    
    # ---------------------------------------------------------
    # 按频率分桶统计并打印
    # ---------------------------------------------------------
    freq_np = go_freq.cpu().numpy()
    rare_mask = freq_np < thresholds[0]
    mid_mask  = (freq_np >= thresholds[0]) & (freq_np < thresholds[1])
    freq_mask = freq_np >= thresholds[1]
    
    print("\n" + "="*50)
    print("原型模块失效诊断报告")
    print("="*50)
    
    print("\n[猜想1验证] 哈达玛积后、归一化前的 L2 范数 (Norm):")
    print(" --> 如果 Rare 的范数极小(如<0.1)且远小于 Freq，说明 Rare 特征已被哈达玛积抹除，归一化只是在放大噪声。")
    print(f"  Rare (n={rare_mask.sum():>4}): {norms_np[rare_mask].mean():.6f}")
    print(f"  Mid  (n={mid_mask.sum():>4}): {norms_np[mid_mask].mean():.6f}")
    print(f"  Freq (n={freq_mask.sum():>4}): {norms_np[freq_mask].mean():.6f}")

    print("\n[猜想2验证] 主网络与动量网络 GO特征 的余弦相似度:")
    print(" --> 如果 Rare 的相似度明显低于 Freq (如<0.8)，说明动量更新严重滞后，发生了原型漂移(Drift)。")
    print(f"  Rare (n={rare_mask.sum():>4}): {drift_sim_np[rare_mask].mean():.4f}")
    print(f"  Mid  (n={mid_mask.sum():>4}): {drift_sim_np[mid_mask].mean():.4f}")
    print(f"  Freq (n={freq_mask.sum():>4}): {drift_sim_np[freq_mask].mean():.4f}")
    print("="*50 + "\n")


# ============================================================
#  类平衡损失：按类别平均，使各类梯度贡献强制 1:1
# ============================================================

def class_balanced_loss(logits, labels, class_ids, valid_mask):
    """
    按类别平均的二分类损失。

    对 batch 中每个被采样的 GO term c：
      1. 选出 class_ids == c 且 valid_mask == 1 的样本
      2. 仅取 logits[:, c] 与 labels[:, c] 计算 BCE
      3. 类内按有效样本数取平均
    最后对所有类取平均，使各类梯度贡献强制 1:1。
    """
    unique_classes = class_ids.unique()
    losses = []

    for c in unique_classes:
        c = c.item()
        mask = (class_ids == c) & (valid_mask > 0.5)
        n = mask.sum().item()
        if n == 0:
            continue

        c_logits = logits[mask, c]          # (n,)
        c_labels = labels[mask, c]          # (n,)
        loss_c = F.binary_cross_entropy_with_logits(c_logits, c_labels)
        losses.append(loss_c)

    if not losses:
        return torch.tensor(0.0, device=logits.device)

    return torch.stack(losses).mean()

def get_support_query_indices(pos_pools):
    support = []
    query = []
    for c in range(len(pos_pools)):
        pos_indices = pos_pools[c]
        if len(pos_indices) >= 2:
            query_set = set(random.sample(pos_indices, len(pos_indices)//2))
            support_set = set(pos_indices) - query_set
            query.append(list(query_set))
            support.append(list(support_set))
        else: # 正样本只有一个
            support.append(pos_indices)
            query.append([])

    return support, query

def load_npy_file(file_path):
    numpy_array = np.load(file_path)
    tensor_data = torch.from_numpy(numpy_array).float()
    return tensor_data

def test_gradient_ratio(model, query, labels, support, support_indices, parent_indices, child_indices, device):
    """用于评估hier loss和分类loss的梯度范数比值，帮助调节权重。
    测试两个 loss 的梯度范数比值，用于调权重。

    分别做两次独立前向+反向传播，记录梯度总范数，
    打印比值 hier_grad_norm / bce_grad_norm。
    目标：让比值接近 1.0。
    """
    model.train()
    q = query.to(device)
    lb = labels.to(device)
    s = support.to(device)
    si = support_indices.to(device)

    # ---- 1. BCE loss 梯度 ----
    model.zero_grad(set_to_none=True)
    logits1, bce_loss, _ = model(query=q, labels=lb, support=s, support_indices=si)
    bce_loss.backward()
    bce_grad_norm = sum(
        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

    # ---- 2. Hier loss 梯度 ----
    model.zero_grad(set_to_none=True)
    logits2, _, _ = model(query=q, labels=lb, support=s, support_indices=si)
    hier_loss = compute_hier_loss(logits2, parent_indices, child_indices)
    hier_loss.backward()
    hier_grad_norm = sum(
        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

    ratio = hier_grad_norm / (bce_grad_norm + 1e-8)

    print(f"[Gradient Ratio Test] "
          f"BCE grad norm={bce_grad_norm:.4f}, "
          f"Hier grad norm={hier_grad_norm:.4f}, "
          f"ratio (hier/bce)={ratio:.4f}")

    return bce_grad_norm, hier_grad_norm, ratio

import numpy as np

def get_go_adjacency_matrices(go_dict):
    """
    根据输入的 GO 关系字典，构建直接父节点邻接矩阵和直接子节点邻接矩阵。
    
    参数:
    go_dict (dict): 格式为 { GO_ID: {'father': [...], 'child': [...], 'ind': int} }
    
    返回:
    father_matrix (np.ndarray): 父节点邻接矩阵 (N x N)
    child_matrix (np.ndarray): 子节点邻接矩阵 (N x N)
    """
    if not go_dict:
        return np.array([]), np.array([])
        
    # 1. 建立 GO ID 到 索引(ind) 的映射关系，并确定矩阵的大小 N
    go_to_ind = {}
    max_ind = -1
    for go_id, info in go_dict.items():
        ind = info['ind']
        go_to_ind[go_id] = ind
        if ind > max_ind:
            max_ind = ind
            
    N = max_ind + 1  # 矩阵的大小由最大索引决定
    
    # 2. 初始化邻接矩阵（全零矩阵）
    father_matrix = np.zeros((N, N), dtype=int)
    child_matrix = np.zeros((N, N), dtype=int)
    
    # 3. 填充邻接矩阵
    for go_id, info in go_dict.items():
        current_ind = info['ind']
        
        # 填充父节点关系
        # 若当前节点 i 的直接父亲是节点 j，则 father_matrix[i][j] = 1
        for father_id in info.get('father', []):
            if father_id in go_to_ind:  # 确保父亲节点在当前字典中存在
                father_ind = go_to_ind[father_id]
                father_matrix[current_ind][father_ind] = 1
                
        # 填充子节点关系
        # 若当前节点 i 的直接孩子是节点 j，则 child_matrix[i][j] = 1
        for child_id in info.get('child', []):
            if child_id in go_to_ind:   # 确保子节点在当前字典中存在
                child_ind = go_to_ind[child_id]
                child_matrix[current_ind][child_ind] = 1
                
    return father_matrix, child_matrix

def get_ancestor_hop_matrix(edges: np.ndarray) -> torch.Tensor:
    """计算每个节点到其祖先节点的最短跳数矩阵。

    Args:
        edges (np.ndarray): 形状为 (num_edges, 2) 的数组，每行为 [parent_index,
          child_index]。

    Returns:
        torch.Tensor: 形状为 (num_nodes, num_nodes) 的张量。
                      matrix[i, j] 表示节点 i 到其祖先节点 j
                      的最短跳数。如果无祖先关系，则为 0。
    """
    if edges.size == 0:
        return torch.empty((0, 0), dtype=torch.long)

    # 1. 确定节点总数（假设节点索引为 0 到 max_idx）
    num_nodes = int(edges.max()) + 1

    # 2. 构建逆向邻接表（从子节点指向其所有直接父节点）
    parent_dict = {i: [] for i in range(num_nodes)}
    for parent, child in edges:
        parent_dict[int(child)].append(int(parent))

    # 3. 初始化跳数矩阵，默认值为 0（表示不可达或无祖先关系）
    hop_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.long)

    # 4. 对每个节点执行 BFS，计算到其所有祖先节点的最短距离
    for start_node in range(num_nodes):
        queue = deque([(start_node, 0)])
        visited = {start_node}  # 避免重复访问

        while queue:
            curr_node, dist = queue.popleft()

            for parent in parent_dict[curr_node]:
                if parent not in visited:
                    visited.add(parent)
                    next_dist = dist + 1
                    # 记录起点到该祖先节点的最短跳数
                    hop_matrix[start_node, parent] = next_dist
                    queue.append((parent, next_dist))

    return hop_matrix


def compute_ancestor_weights(hop_counts: torch.Tensor, class_counts: torch.Tensor, lambda_: float) -> torch.Tensor:
    """计算每个 GO term 节点对其所有祖先节点的参考权重矩阵。

    权重公式: γ * n / (n + λ)

    - γ = 1 / d：距离衰退系数，d 为节点 i 到祖先 j 的最短跳数。
      距离越近的祖先语义越相关，γ 越大，权重越高。
    - n = class_counts[j]：祖先节点 j 注释到的蛋白质样本数量。
      样本量越大的祖先越可靠，权重越高。
    - λ：超参数，控制样本量对权重的影响程度。
      λ 越大，样本量差异对权重的影响越小。

    Args:
        hop_counts:  形状 (num_nodes, num_nodes)。
                     hop_counts[i][j] > 0 表示 j 是 i 的祖先，值为最短跳数 d；
                     hop_counts[i][j] == 0 表示 j 不是 i 的祖先。
        class_counts: 形状 (num_nodes,)，每个 GO term 注释到的蛋白质样本数量 n。
        lambda_:     超参数 λ，调节样本数量的影响。

    Returns:
        weight: 形状 (num_nodes, num_nodes)。
                weight[i][j] 表示节点 i 对其祖先节点 j 的参考权重。
                若 j 不是 i 的祖先，weight[i][j] = 0。

    Example:
        >>> # 节点 0 的祖先: 节点 2 (跳数 1, 样本数 100), 节点 3 (跳数 3, 样本数 10)
        >>> # weight[0][2] = (1/1) * 100/(100+λ)
        >>> # weight[0][3] = (1/3) * 10/(10+λ)
    """
    ancestor_mask = hop_counts > 0

    # γ = 1 / d，非祖先位置自动为 0
    gamma = torch.where(ancestor_mask, 1.0 / hop_counts.float(), torch.tensor(0.0))

    # n / (n + λ)，样本量缩放因子（广播到所有行）
    n = class_counts.float()
    sample_factor = n / (n + lambda_)

    weight = gamma * sample_factor.unsqueeze(0)

    return weight


def compute_ic(go2id, train_seq_data, obo_path):
    """基于训练集标注计算每个 GO term 的信息量 (IC)。

    使用 goatools 的 TermCounts 沿 DAG 传播标注计数，
    IC(go) = -log(freq(go))，freq 为传播后的归一化频率。
    根节点（最泛化）IC≈0，越具体的 term IC 越大。

    Args:
        go2id:        dict[GO_ID -> {'ind': int, 'father': [...], 'child': [...]}]
        train_seq_data: list[dict]，每个元素含 'label' 字段（GO term 类别索引列表）
        obo_path:     go-basic.obo 文件路径

    Returns:
        ic: torch.Tensor, 形状 (num_classes,)，按 go2id 的 ind 索引排列
    """
    from goatools.obo_parser import GODag
    from goatools.semantic import TermCounts

    # 类别索引 -> GO ID
    idx2go = {v['ind']: k for k, v in go2id.items() if isinstance(v, dict) and 'ind' in v}

    # 构建 protein_id -> set(GO IDs)，TermCounts 要求此格式
    gene2gos = {}
    for i, item in enumerate(train_seq_data):
        gos = {idx2go[idx] for idx in item.get('label', []) if idx in idx2go}
        if gos:
            gene2gos[i] = gos

    godag = GODag(obo_path, optional_attrs={'relationship'})
    tcnt = TermCounts(godag, gene2gos)

    ic = np.zeros(len(go2id), dtype=np.float64)
    for go_id, info in go2id.items():
        if isinstance(info, dict) and 'ind' in info:
            freq = tcnt.get_term_freq(go_id)
            ic[info['ind']] = -math.log(freq) if freq > 0 else 0.0

    return torch.from_numpy(ic).float()


def compute_ancestor_weights_ic(hop_counts: torch.Tensor, ic: torch.Tensor,
                                class_counts: torch.Tensor, lambda_: float,
                                beta_: float = 1.0) -> torch.Tensor:
    """基于信息量 (IC) 计算每个 GO term 节点对其祖先节点的参考权重矩阵。

    权重公式: exp(-β * max(ΔIC, 0)) * n / (n + λ)

    - ΔIC = IC(ancestor) - IC(node)：祖先与当前节点的信息量差。
      ΔIC <= 0 表示祖先更具体（罕见），权重为 1（不衰减）；
      ΔIC > 0 表示祖先更泛化（常见），IC 差越大权重越低。
      相比跳数，IC 直接反映生物学特异性，泛化程度高的祖先权重低。
    - n = class_counts[j]：祖先节点 j 的正样本数，样本量越大越可靠。
    - λ：控制样本量对权重的影响程度。
    - β：控制 IC 差对权重的衰减速度，β 越大对泛化祖先惩罚越强。

    Args:
        hop_counts:  形状 (num_nodes, num_nodes)。
                     hop_counts[i][j] > 0 表示 j 是 i 的祖先（仅用作祖先关系掩码）。
        ic:          形状 (num_nodes,)，每个 GO term 的信息量。
        class_counts: 形状 (num_nodes,)，每个 GO term 的正样本数。
        lambda_:     超参数 λ，调节样本数量的影响。
        beta_:       超参数 β，调节 IC 差的衰减速度。

    Returns:
        weight: 形状 (num_nodes, num_nodes)。
                weight[i][j] 表示节点 i 对其祖先节点 j 的参考权重。
                若 j 不是 i 的祖先，weight[i][j] = 0。
    """
    ancestor_mask = hop_counts > 0  # (num_nodes, num_nodes)

    # ΔIC[i, j] = IC(j) - IC(i)，广播: ic[j] 按列, ic[i] 按行
    ic_diff = ic.unsqueeze(1) - ic.unsqueeze(0)  # (num_nodes, num_nodes)
    # 仅惩罚更泛化的祖先 (ΔIC > 0)，更具体的祖先不衰减
    ic_factor = torch.exp(-beta_ * ic_diff.clamp(min=0.0))

    # n / (n + λ)，样本量缩放因子（按列广播，对应祖先 j）
    n = class_counts.float()
    sample_factor = n / (n + lambda_)

    weight = ic_factor * sample_factor.unsqueeze(0)
    weight = weight * ancestor_mask.float()  # 非祖先位置置 0

    return weight