import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端以支持动态更新
import matplotlib.pyplot as plt
import numpy as np

class PCN_Visual:
    def __init__(self, depth, device="cpu", fixed_xlim=(-200, 200), fixed_ylim=(-200, 200)):
        """
        :param depth: 网络层数
        :param device: 当前设备（仅记录，不影响绘图）
        """
        self.depth = depth
        self.device = device
        self.fixed_xlim = fixed_xlim
        self.fixed_ylim = fixed_ylim

        # 初始化数据容器
        self.x_list = [np.zeros((1, 2)) for _ in range(depth)]
        self.v_list = [np.zeros((1, 2)) for _ in range(depth)]

        # 创建绘图窗口
        self.fig, self.axs = plt.subplots(1, depth, figsize=(4 * depth, 4))
        if depth == 1:
            self.axs = [self.axs]  # 保证统一索引方式

        for i, ax in enumerate(self.axs):
            ax.set_title(f"Layer {i}")
            ax.set_xlabel("x[0]")
            ax.set_ylabel("x[1]")
            ax.grid(True)

        # 方便动态更新
        plt.ion()
        plt.show()

    def set_x(self, x_list):
        """更新每层状态向量 x"""
        self.x_list = [x.detach().cpu().numpy().reshape(1, -1) for x in x_list]

    def set_v(self, v_list):
        """更新每层预测向量 v"""
        self.v_list = [v.detach().cpu().numpy().reshape(1, -1) for v in v_list]

    def visualize(self, phase="Prediction", rounds=0):
        """
        动态可视化
        :param phase: 当前阶段（Prediction / Training）
        :param rounds: 当前迭代轮数
        """
        for i, ax in enumerate(self.axs):
            ax.cla()  # 清空子图
            ax.set_title(f"{phase} | Layer {i} | iter={rounds}")
            ax.set_xlabel("x[0]")
            ax.set_ylabel("x[1]")
            ax.grid(True)

            # **固定坐标轴范围**
            if self.fixed_xlim is not None:
                ax.set_xlim(*self.fixed_xlim)
            if self.fixed_ylim is not None:
                ax.set_ylim(*self.fixed_ylim)

            # 绘制真实状态 x[i]
            x = self.x_list[i]
            ax.scatter(x[:, 0], x[:, 1], color="blue", label="x", s=50, marker="o")

            # 绘制预测值 v[i]
            v = self.v_list[i]
            ax.scatter(v[:, 0], v[:, 1], color="red", label="v", s=50, marker="x")

            # 画出 x 到 v 的连线
            ax.plot([x[0, 0], v[0, 0]], [x[0, 1], v[0, 1]], color="gray", linestyle="--")

            ax.legend()

        plt.tight_layout()
        plt.pause(0.05)
