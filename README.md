<div style="text-align: center;" align="center">

# `amplify_bbopt_ext`

Fixstars Amplify BBOptに基づいた，ブラックボックス離散最適化問題のための拡張機能とサンプルコード集．

**Links:** [スタートガイド](#スタートガイド)
— [インストール方法](#インストール)
— [サンプルコード集](./examples/)

</div>

---

このライブラリでは，ブラックボックス離散最適化問題のために研究・開発した以下の最適化手法をAmplify BBOptに基づいて実装しています．

- [SWIFT-FMQA](./examples/fmqa_latest.ipynb)
- [LHS/Sobol'-FMQA](./examples/lhs_sobol_fmqa.ipynb)
- （今後も追加予定）

国立研究開発法人産業技術総合研究所量子・AI融合技術グローバル研究センター（G-QuAT）の保有する量子・古典融合計算基盤ABCI-Qのコンピューティングリソースを活用して，研究成果に基づく最適化手法を検証することができます．

---

## スタートガイド

ABCI-Qの[Open OnDemand](https://g-quat-abciq.github.io/abciq-docs/ja/open-ondemand/)から起動できる対話型開発環境Jupyter Lab上で最適化手法を試すことができます．使用感を示すために，以下に簡単なコード例を紹介します．

```python
import amplify_bbopt
from amplify_bbopt import (
    DiscretizationSpec,
    RealVariable,
    blackbox,
)
from amplify import FixstarsClient
from amplify_bbopt_ext.runner import run

from datetime import timedelta
import numpy as np


###############################################################################
# ブラックボックス関数の定義
###############################################################################
# ブラックボックス関数としてRastrigin関数を定義する
# cf. https://amplify.fixstars.com/ja/docs/amplify-bbopt/v1/example_rastrigin.html#id2
def rastrigin_function(x: list[float]) -> float:
    """A test function, yieldng the global min of 0 at (0, 0, ...).

    Args:
        x (list[float]): An input vector.

    Returns:
        float: The function value.
    """
    return (
        10 * len(x)
        + (np.array(x) ** 2 - 10 * np.cos(2 * np.pi * np.array(x))).sum()
    )


# 問題・条件設定
rng = np.random.default_rng()
num_vars = 5  # 問題次元 (実数決定変数数)
var_min = -3.0  # 実数決定変数の最小値
var_max = 3.0  # 実数決定変数の最大値
disc_spec = DiscretizationSpec(num_bins=101)  # 離散点数を 101 に設定

# 実数決定変数リストの作成
x_list = [
    RealVariable((var_min, var_max), discretization_spec=disc_spec)
    for _ in range(num_vars)
]


# ブラックボックス関数の定義
@blackbox
def func(input_x: list[float] = x_list) -> float:  # type: ignore
    return rastrigin_function(input_x)

###############################################################################
# 最適化の実行
###############################################################################
# Amplify AEの準備
client = FixstarsClient()
# Fixstars Amplify Engineのurlを設定
client.url = "http://localhost:1080"
client.parameters.timeout = timedelta(milliseconds=1000)  # 1 second
# 使用する GPU 数を 4 に設定
client.parameters.num_gpus = 4

# パラメータ設定（詳細はコード例を参照）
size_limit = 200
K = 8
n_iter = 200
epochs = 1000
optimizer_params = {"lr": 0.01}
lr_scheduler_class = None
seed = None

# 結果を取得
result = run(
    client,
    func,
    size_limit,
    K,
    n_iter,
    initial_samples,
    epochs,
    optimizer_params,
    lr_scheduler_class,
)

###############################################################################
# 結果の可視化
###############################################################################
import matplotlib.pyplot as plt
import numpy as np

# 最適化履歴からこれまでの最小値（Best Objective Value）の推移を計算
y_history = result.training_data.y
best_y = np.minimum.accumulate(y_history)

fig, ax = plt.subplots(figsize=(10, 6))
n_initial = len(initial_samples)

# 評価回数（x軸）は 1 からスタート
x_evals = np.arange(1, len(best_y) + 1)

# 全サンプリング点の散布図
ax.scatter(x_evals, y_history, color='tab:blue', alpha=0.3, label='Sampled Points', s=20)

# これまでの最小値の推移の折れ線
ax.plot(x_evals, best_y, linestyle='-', color='tab:red', label='Best Objective Value')

# フェーズの境界線
ax.axvline(x=n_initial, color='k', linestyle='--', alpha=0.5, label='End of Initial Data')

# 背景を色分けしてフェーズを区別
ax.axvspan(0, n_initial, alpha=0.1, color='gray', label='Initial Sampling Phase')
ax.axvspan(n_initial, len(best_y) + 1, alpha=0.05, color='green', label='FMQA Optimization Phase')

ax.set_xlabel('Number of evaluations', fontsize=16)
ax.set_ylabel('Objective value', fontsize=16)
ax.set_title('Optimization History on Rastrigin function', fontsize=18)
ax.legend(fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
```

## インストール

```bash
python3 -m pip install git+https://github.com/Shu-Tanaka-Group/amplify-bbopt-ext.git

```

## 謝辞

## NEDO（FY2023–FY2025）

This library is based on results obtained from a project, JPNP23003, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
