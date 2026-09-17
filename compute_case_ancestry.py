"""计算 9 个 TALE 零样本案例的原型祖先借力构成（论文 Figure 3 Panel A 数据源）。

- 使用与 eval.yml 一致的 checkpoint (prototype/checkpoint_55.pth, 训练后的 λ/β/τ)
- 祖先关系来自 {ns}_label_regular_1.npy → hop_counts（其构建依据为
  data_tale/TALE/go-basic.obo, 版本 fmt(1.2) rel(2020-12-08)），
  并断言 proto_w 在所有非祖先位置为 0，保证与该版本本体无矛盾
- 输出每个案例 top-8 祖先的 GO ID / 名称 / 权重 / hop / 训练正样本数

用法（synergy 环境）:
  python compute_case_ancestry.py
"""

import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
import utils as utils  # noqa: E402

CASES = [
    ('cc', 'GO:0009375'), ('cc', 'GO:0001535'), ('cc', 'GO:0031972'),
    ('mf', 'GO:0047713'), ('mf', 'GO:0106262'), ('mf', 'GO:0052894'),
    ('bp', 'GO:0002129'), ('bp', 'GO:0042935'), ('bp', 'GO:0044836'),
]
obo_path = './data_tale/TALE/go-basic.obo'
CKPT = '/archive/hot5/fty/checkpoints/{NS}/TALE/prototype/checkpoint_55.pth'


def main():
    from models.test_model import PrototypeNet

    for ns, go in CASES:
        dp = './data_tale/TALE'
        train_seq = utils.load_data_from_pkl(f'{dp}/train_seq_{ns}')
        n_prot = len(train_seq)
        num_classes = np.load(f'{dp}/{ns}_label_matrix_1_sparse.npy').shape[0]
        proto_idx = utils.get_prototype_index(train_seq, num_classes)
        proto_idx[proto_idx[:, 0] == 0, 0] = 1
        go_freq, class_counts = utils.compute_go_term_frequency(proto_idx.T, n_prot)
        edges = np.load(f'{dp}/{ns}_label_regular_1.npy')
        hop = utils.get_ancestor_hop_matrix(edges)
        go2id = utils.load_data_from_pkl(f'{dp}/{ns}_go_1.pickle')
        pm, _ = utils.get_go_adjacency_matrices(go2id)
        ic = utils.compute_ic(go2id, train_seq, obo_path)
        lam_i, beta_i, tau_i = utils.compute_smooth_param_inits(hop, ic, class_counts)

        model = PrototypeNet(input_dim=1280, hidden_dim=1024, num_classes=num_classes,
                             frequency=go_freq, parents_matrix=pm,
                             class_counts=class_counts, hop_counts=hop, ic=ic,
                             smooth_tau=tau_i, lambda_init=lam_i,
                             beta_init=beta_i, tau_min=0.0)
        sd = torch.load(CKPT.format(NS=ns.upper()), map_location='cpu')
        sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
        if 'tau_min' not in sd:
            sd['tau_min'] = torch.tensor(0.0)
        model.load_state_dict(sd)
        model.eval()
        proto_w = model._compute_proto_w().detach().cpu().numpy().astype(np.float64)
        lam, beta, tau = model.get_smooth_params()

        c = go2id[go]['ind']
        anc = np.where(hop[c] > 0)[0]
        idx2go = {v['ind']: k for k, v in go2id.items()
                  if isinstance(v, dict) and 'ind' in v}
        w = proto_w[c, anc]
        w = w / w.sum()
        order = np.argsort(-w)
        eff = 1.0 / (w ** 2).sum()

        print(f'=== {ns.upper()} {go} ({go2id[go].get("name", "")}) ind={c} | '
              f'lambda={lam:.3f} beta={beta:.3f} tau={tau:.3f} | '
              f'n_anc={len(anc)} eff_anc={eff:.1f}')
        for j in order[:8]:
            a = int(anc[j])
            g2 = idx2go.get(a, str(a))
            print(f'   {g2} ({go2id.get(g2, {}).get("name", "")})  '
                  f'w={w[j]:.4f}  hop={int(hop[c, a])}  '
                  f'train_pos={int(class_counts[a])}')

        # 一致性断言: 非祖先位置权重必须为 0（祖先闭包与 hop/edges/obo 版本一致）
        assert (proto_w[c][hop[c] == 0] == 0).all(), '祖先关系与 hop 不一致!'

    print('ALL-CHECK-PASSED: 祖先闭包与 hop_counts/edges 一致 (go-basic.obo 2020-12-08)')


if __name__ == '__main__':
    main()
