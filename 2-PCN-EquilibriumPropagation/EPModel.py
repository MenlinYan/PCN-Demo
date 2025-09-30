import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# EP 模型 (简化版)
# --------------------------
class EPModel(nn.Module):
    def __init__(self, dims, beta=1e-3, n_iter=20, lr_state=0.5, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.dims = dims
        self.beta = beta
        self.n_iter = n_iter
        self.lr_state = lr_state

        # 权重参数（对称约束可选）
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(d_in, d_out) * 0.1)
            for d_in, d_out in zip(dims[:-1], dims[1:])
        ])

    def energy(self, states):
        """能量函数 E = 0.5 * sum ||s_{k+1} - s_k W||^2"""
        E = torch.tensor(0.0, device=states[0].device)
        for i, W in enumerate(self.weights):
            pred = states[i] @ W
            E = E + 0.5 * F.mse_loss(states[i+1], pred, reduction="sum")
        return E

    def relax(self, x, y=None, beta=0.0):
        """状态松弛: free phase 或 weakly clamped phase"""
        # 初始化 states（前向初始化）
        states = [x]
        for W in self.weights:
            states.append(states[-1] @ W)

        # 迭代更新隐藏层和输出层
        for _ in range(self.n_iter):
            E = self.energy(states)
            if beta != 0.0 and y is not None:
                E = E + beta * 0.5 * F.mse_loss(states[-1], y, reduction="sum")
            grads = torch.autograd.grad(E, states[1:], create_graph=False)
            with torch.no_grad():
                for i, g in enumerate(grads, start=1):
                    states[i] = (states[i] - self.lr_state * g).detach().requires_grad_(True)
        return [s.detach() for s in states]

    def compute_grads(self, x, y):
        """返回 EP 梯度"""
        # free phase
        s0 = self.relax(x, y, beta=0.0)
        # weakly clamped phase
        sb = self.relax(x, y, beta=self.beta)

        # 计算 dF/dW
        E0 = self.energy(s0)
        Eb = self.energy(sb) + self.beta * 0.5 * F.mse_loss(sb[-1], y, reduction="sum")

        g0 = torch.autograd.grad(E0, self.weights, retain_graph=False)
        gb = torch.autograd.grad(Eb, self.weights, retain_graph=False)

        grads = [(gb_i - g0_i) / self.beta for gb_i, g0_i in zip(gb, g0)]
        return grads, F.mse_loss(s0[-1], y).item()
    
    def update(self, grads, lr):
        """手动更新权重"""
        with torch.no_grad():
            for W, g in zip(self.weights, grads):
                W -= lr * g



# --------------------------
# 测试运行
# --------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(10, 2)
    W_true = torch.tensor([[2., 0.], [0., 2.]])
    y = x @ W_true
    dataloader = [(x, y)]
    epochs = 30

    ep_model = EPModel([2, 2, 2], beta=1e-3, n_iter=30, seed=42)

    for epoch in range(epochs):
        # 外部负责取 batch 和更新模型
        for x_batch, y_batch in dataloader:
            # 更新 EP 模型
            ep_grads, ep_loss = ep_model.compute_grads(x_batch, y_batch)
            ep_model.update(ep_grads, lr=0.01)
            print("epoch", epoch, "ep_loss:", ep_loss)
