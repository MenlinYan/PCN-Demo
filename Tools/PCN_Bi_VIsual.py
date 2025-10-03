import matplotlib
matplotlib.use('TkAgg')  # TkAgg 后端支持动态更新
import matplotlib.pyplot as plt
import numpy as np

class PCN_Bi_Visual:
    def __init__(self, depth, device="cpu", fixed_xlim=(-200, 200), fixed_ylim=(-200, 200)):
        """
        BiPCN 可视化工具
        :param depth: 网络层数
        :param device: 当前设备（仅记录，不影响绘图）
        """
        self.depth = depth
        self.device = device
        self.fixed_xlim = fixed_xlim
        self.fixed_ylim = fixed_ylim

        # 初始化容器
        self.x_list = [np.zeros((1, 2)) for _ in range(depth)]
        self.vf_list = [np.zeros((1, 2)) for _ in range(depth)]  # forward prediction
        self.vb_list = [np.zeros((1, 2)) for _ in range(depth)]  # backward prediction

        # 创建子图
        self.fig, self.axs = plt.subplots(1, depth, figsize=(4 * depth, 4))
        if depth == 1:
            self.axs = [self.axs]

        for i, ax in enumerate(self.axs):
            ax.set_title(f"Layer {i}")
            ax.set_xlabel("x[0]")
            ax.set_ylabel("x[1]")
            ax.grid(True)

        plt.ion()
        plt.show()

    def set_x(self, x_list):
        """更新每层状态向量 x"""
        self.x_list = [x.detach().cpu().numpy().reshape(1, -1) for x in x_list]

    def set_vf(self, vf_list):
        """更新每层前向预测向量 v_forward"""
        self.vf_list = [v.detach().cpu().numpy().reshape(1, -1) for v in vf_list]

    def set_vb(self, vb_list):
        """更新每层后向预测向量 v_backward"""
        self.vb_list = [v.detach().cpu().numpy().reshape(1, -1) for v in vb_list]

    def visualize(self, phase="Prediction", rounds=0):
        """
        动态可视化
        :param phase: 当前阶段（Prediction / Training）
        :param rounds: 当前迭代次数
        """
        for i, ax in enumerate(self.axs):
            ax.cla()
            ax.set_title(f"{phase} | Layer {i} | iter={rounds}")
            ax.set_xlabel("x[0]")
            ax.set_ylabel("x[1]")
            ax.grid(True)

            # 固定坐标
            if self.fixed_xlim is not None:
                ax.set_xlim(*self.fixed_xlim)
            if self.fixed_ylim is not None:
                ax.set_ylim(*self.fixed_ylim)

            # 状态向量 x
            x = self.x_list[i]
            ax.scatter(x[:, 0], x[:, 1], color="blue", label="x", s=60, marker="o")

            # 前向预测 v_f
            vf = self.vf_list[i]
            ax.scatter(vf[:, 0], vf[:, 1], color="red", label="v_forward", s=60, marker="x")
            ax.plot([x[0, 0], vf[0, 0]], [x[0, 1], vf[0, 1]], color="red", linestyle="--")

            # 后向预测 v_b
            vb = self.vb_list[i]
            ax.scatter(vb[:, 0], vb[:, 1], color="green", label="v_backward", s=60, marker="^")
            ax.plot([x[0, 0], vb[0, 0]], [x[0, 1], vb[0, 1]], color="green", linestyle=":")

            ax.legend()

        plt.tight_layout()
        plt.pause(0.05)
