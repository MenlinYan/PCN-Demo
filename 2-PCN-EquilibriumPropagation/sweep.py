import pandas as pd
import torch
import itertools
import numpy as np
from matplotlib import pyplot as plt

from BPModel import BPModel
from EPModel import EPModel
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'

def cosine_similarities(bp_grads, ep_grads):
    sims = []
    for g_bp, g_ep in zip(bp_grads, ep_grads):
        sim = torch.nn.functional.cosine_similarity(
            g_bp.flatten(), g_ep.flatten(), dim=0
        ).item()
        sims.append(sim)
    return sims


def run_experiment(beta, n_iter, lr_state, n_epochs=5):
    torch.manual_seed(42)
    # toy 数据
    # x = torch.randn(10, 2)
    x = torch.tensor([[1., 1.], [2., 3.], [4., 5.], [3., 1.], [2., 5.]])
    W_true = torch.tensor([[2., 0.], [0., 2.]])
    y = x @ W_true
    dataloader = [(x, y)]

    # 初始化模型
    bp_model = BPModel([2, 4, 2], seed=42)
    ep_model = EPModel([2, 4, 2], beta=beta, n_iter=n_iter, lr_state=lr_state, seed=42)

    similarities_all = []

    for epoch in range(n_epochs):
        for x_batch, y_batch in dataloader:
            # BP
            bp_grads, bp_loss = bp_model.compute_grads(x_batch, y_batch)
            bp_model.update(bp_grads, lr=0.01)

            # EP
            ep_grads, ep_loss = ep_model.compute_grads(x_batch, y_batch)
            ep_model.update(ep_grads, lr=0.01)

            sims = cosine_similarities(bp_grads, ep_grads)
            similarities_all.append(np.mean(sims))

    return np.mean(similarities_all)


if __name__ == "__main__":
    # 超参数 sweep 范围
    beta_list = [1e-6, 1e-3, 1e-1]
    n_iter_list = [1, 5, 10, 20]
    lr_state_list = [1e-3, 1e-2, 5e-2, 0.1, 0.2, 0.5]

    results = []
    for beta, n_iter, lr_state in itertools.product(beta_list, n_iter_list, lr_state_list):
        avg_sim = run_experiment(beta, n_iter, lr_state, n_epochs=5)
        results.append((beta, n_iter, lr_state, avg_sim))

    # 转为 DataFrame
    df = pd.DataFrame(results, columns=["beta", "n_iter", "lr_state", "avg_sim"])

    # 可视化：针对不同 beta 画 3D 曲面图
    fig = plt.figure(figsize=(15, 5))

    for i, beta in enumerate(beta_list, 1):
        ax = fig.add_subplot(1, len(beta_list), i, projection='3d')
        subset = df[df["beta"] == beta]
        X, Y = np.meshgrid(sorted(n_iter_list), sorted(lr_state_list))
        Z = np.zeros_like(X, dtype=float)
        for ix, n_iter in enumerate(sorted(n_iter_list)):
            for iy, lr_state in enumerate(sorted(lr_state_list)):
                Z[iy, ix] = subset[(subset["n_iter"] == n_iter) & (subset["lr_state"] == lr_state)]["avg_sim"].values[0]
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title(f"beta={beta}")
        ax.set_xlabel("n_iter")
        ax.set_ylabel("lr_state")
        ax.set_zlabel("avg_sim")

    plt.tight_layout()
    plt.show()