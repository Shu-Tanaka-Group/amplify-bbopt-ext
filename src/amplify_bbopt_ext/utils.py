import matplotlib.pyplot as plt
import numpy as np


def plot_optimization_history(result, num_initial_data, *, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    # 最適化履歴からこれまでの最小値（Best Objective Value）の推移を計算
    y_history = result.training_data._y
    best_y = np.minimum.accumulate(y_history)

    # 評価回数（x軸）は 1 からスタート
    x_evals = np.arange(1, len(best_y) + 1)

    # 全サンプリング点の散布図
    ax.scatter(x_evals, y_history, color='tab:blue', alpha=0.3, label='Sampled Points', s=20)

    # これまでの最小値の推移の折れ線
    ax.plot(x_evals, best_y, linestyle='-', color='tab:red', label='Best Objective Value', markersize=6)

    # フェーズの境界線
    ax.axvline(x=num_initial_data, color='k', linestyle='--', alpha=0.5, label='End of Initial Data')

    # 背景を色分けしてフェーズを区別
    ax.axvspan(0, num_initial_data, alpha=0.1, color='gray', label='Initial Sampling Phase')
    ax.axvspan(num_initial_data, len(best_y) + 1, alpha=0.05, color='green', label='FMQA Optimization Phase')

    ax.set_xlabel('Number of evaluations', fontsize=16)
    ax.set_ylabel('Objective value', fontsize=16)
    ax.set_title('Optimization History', fontsize=18)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)

    return fig, ax


