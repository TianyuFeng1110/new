"""案例研究绘图（论文 Figure 3，方案 A 简化版：仅 Panel B，两步式）。

工作流（数据审核与绘图分离，样式可自行在 Excel 中调整）:
  步骤 1  JSON → Excel:  python evaluate/plot_case_figure.py --json <records.json> --to-excel <out.xlsx>
          每个案例一行，包含全部字段；可在 Excel 中自行修改显示名称、删除行、调整顺序。
  步骤 2  Excel → 图:    python evaluate/plot_case_figure.py --excel <out.xlsx> --out <png>
          读取 Excel 中当前内容绘制 Panel B（Proto vs MLP 零样本类内排名对照条形图）。
          Excel 中标记删除（如删除该行或 Display=0）的案例不会出现在图中。

Excel 列说明（绘图读取前四列 + 可选 Display 列）:
  - Display:   1=画入图中, 0=跳过（默认 1）
  - Case:      案例显示名（可自由改，如缩短的 GO 名称）
  - ProtoRank: 原型模块的零样本类内排名（筛选保证为 1）
  - MLPRank:   骨干模块的零样本类内排名
  - 其余列（GO/Name/TestPos/...）仅供参考，不参与绘图。

依赖: pandas + openpyxl（Excel 写读）+ matplotlib。无需 torch。
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GREEN = '#2ca02c'
GRAY = '#9e9e9e'


# ============================================================
#  步骤 1: JSON → Excel
# ============================================================

def json_to_excel(json_path, excel_path):
    import pandas as pd

    with open(json_path, encoding='utf-8') as f:
        records = json.load(f)
    if not records:
        raise SystemExit('记录为空')

    rows = []
    for rec in records:
        name = rec.get('name') or ''
        if not name or name == '(name 不可用)':
            name = rec['go']
        rows.append({
            'Display': 1,                                  # 改 0 可从图中剔除该行
            'Case': name,                                  # 显示名，可自行修改
            'ProtoRank': 1,                                # 筛选标准保证 Proto top-1 命中
            'MLPRank': rec['mlp_rank'],
            'GO': rec['go'],
            'FullName': name,
            'TestPos': rec['test_pos'],
            'ProteinIdx': rec['protein_idx'],
            'ProtoScore': round(rec['proto_score'], 6),
            'MLPScore': round(rec['mlp_score'], 6),
            'ProtoMargin': round(rec['proto_margin'], 6),
            'EffAncestors': round(rec['eff_ancestors'], 1),
            'TopAncestors': ' | '.join(
                f"{a['go']}({a['name']}) w={a['weight']:.3f} hop={a['hop']} n={a['train_pos']}"
                for a in rec.get('ancestors', [])),
        })

    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False)
    print(f'✔ Excel 已保存: {excel_path} ({len(df)} 个案例)')
    print('  可在 Excel 中修改 Case 列(显示名)、Display 列(0=不画入图)、'
          '行顺序(图中自上而下顺序)，保存后再执行步骤 2。')


# ============================================================
#  步骤 2: Excel → Panel B 图
# ============================================================

def excel_to_figure(excel_path, out_path, n_zs_classes=None, title=None):
    import pandas as pd

    df = pd.read_excel(excel_path)
    if 'Display' in df.columns:
        df = df[df['Display'] == 1]
    if df.empty:
        raise SystemExit('Excel 中没有 Display=1 的行，无法绘图')

    labels = df['Case'].astype(str).tolist()
    proto_ranks = df['ProtoRank'].astype(int).tolist()
    mlp_ranks = df['MLPRank'].astype(int).tolist()
    n_zs = n_zs_classes or max(mlp_ranks)

    y = np.arange(len(df))[::-1]
    h = 0.34

    fig, ax = plt.subplots(figsize=(8, max(2.4, 0.9 * len(df) + 1.2)))
    ax.barh(y + h / 2, proto_ranks, height=h, color=GREEN, alpha=0.9,
            label='Prototype (ours)')
    ax.barh(y - h / 2, mlp_ranks, height=h, color=GRAY, alpha=0.85,
            label='Discriminative backbone (MLP)')

    for yi, pr, mr in zip(y, proto_ranks, mlp_ranks):
        ax.text(pr + n_zs * 0.012, yi + h / 2, f'rank {pr}', va='center',
                fontsize=8.5, color=GREEN, fontweight='bold')
        ax.text(mr + n_zs * 0.012, yi - h / 2, f'rank {mr}', va='center',
                fontsize=8.5, color='#555555')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f'Rank within the zero-shot candidate set (size = {n_zs})',
                  fontsize=10)
    ax.set_xlim(0, max(mlp_ranks) * 1.18)
    ax.legend(loc='lower right', fontsize=9)
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.35)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    for ext in ('png', 'pdf'):
        p = f'{os.path.splitext(out_path)[0]}.{ext}'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'✔ 已保存 {p}')
    plt.close(fig)


# ============================================================
#  主流程（两步式）
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Case-study figure, two-step workflow (JSON → Excel → figure)')
    parser.add_argument('--json', type=str,
                        help='步骤 1: eval_zero_shot_case_study 保存的 records JSON')
    parser.add_argument('--to-excel', type=str, dest='to_excel',
                        help='步骤 1: 输出 Excel 路径')
    parser.add_argument('--excel', type=str,
                        help='步骤 2: (可编辑后的) Excel 路径')
    parser.add_argument('--out', type=str, default='./tmp/fig3/fig3_case_study.png',
                        help='步骤 2: 输出图片路径 (png/pdf 同名双格式)')
    parser.add_argument('--n-zs-classes', type=int, default=None,
                        help='零样本候选类总数（横轴标注；建议显式传入，如 TALE/CC=58）')
    parser.add_argument('--title', type=str, default=None,
                        help='可选图标题')
    args = parser.parse_args()

    if args.json and args.to_excel:
        json_to_excel(args.json, args.to_excel)
    elif args.excel:
        excel_to_figure(args.excel, args.out,
                        n_zs_classes=args.n_zs_classes, title=args.title)
    else:
        parser.print_help()
        print('\n两种用法:\n'
              '  步骤 1: --json <records.json> --to-excel <out.xlsx>\n'
              '  步骤 2: --excel <out.xlsx> --out <fig.png> [--n-zs-classes 58]')


if __name__ == '__main__':
    main()
