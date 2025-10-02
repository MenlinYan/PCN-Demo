import torch
import torch.nn as nn
import torch.nn.functional as F

class BiPCN(nn.Module):
    def __init__(self, dims, device="cpu"):
        super(BiPCN, self).__init__()
        self.dims = dims
        self.n_layers = len(dims)

        # 前向 (bottom-up) 权重 V: x_{l} -> x_{l+1}
        self.V = nn.ParameterList([
            nn.Parameter(torch.randn(d_in, d_out) * 0.1)
            for d_in, d_out in zip(self.dims[:-1], self.dims[1:])
        ])

        # 后向 (top-down) 权重 W: x_{l+1} -> x_{l}
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(d_out, d_in) * 0.1)
            for d_in, d_out in zip(self.dims[:-1], self.dims[1:])
        ])

        self.optim_params = torch.optim.Adam(
            list(self.V) + list(self.W), lr=0.01
        )

    def compute_energy(self, x_list):
        energy = 0
        # 遍历层间连接
        for l in range(self.n_layers - 1):
            # top-down 预测误差
            v_down = x_list[l+1] @ self.W[l]
            energy += F.mse_loss(v_down, x_list[l], reduction='mean')
            # bottom-up 预测误差
            v_up = x_list[l] @ self.V[l]
            energy += F.mse_loss(v_up, x_list[l+1], reduction='mean')
        return energy

    def inference(self, x, y=None, n_iter=10, lr=0.1):
        """双向推理，优化隐变量"""
        x_list = [x]
        # 初始化：用前向权重 V 前馈
        for l in range(self.n_layers - 1):
            x_next = x_list[-1] @ self.V[l]
            x_list.append(x_next.clone().detach().requires_grad_(True))

        # 如果有监督目标 y，则固定最后一层
        if y is not None:
            x_list[-1] = y.detach()
            x_list[-1].requires_grad = False

        begin, end = 1, self.n_layers if y is None else self.n_layers - 1

        # 推理循环
        for it in range(n_iter):
            E = self.compute_energy(x_list)
            for xvar in x_list[begin:end]:
                if xvar.grad is not None: xvar.grad.zero_()
            E.backward(retain_graph=True)
            for xvar in x_list[begin:end]:
                xvar.data -= lr * xvar.grad
            if it % 10 == 0 or it == n_iter - 1:
                print(f'Inference Iter {it}, Energy: {E.item():.4f}')
        return x_list

    def train_step(self, x, y, n_iter=20, lr_x=0.1):
        # 隐变量推理
        x_list = self.inference(x, y, n_iter=n_iter, lr=lr_x)
        # 损失
        E = self.compute_energy(x_list)
        # 权重更新
        self.optim_params.zero_grad()
        E.backward()
        self.optim_params.step()
        return E.item()

    def fit(self, train_loader, batch_size=32, n_epochs=10, n_iter=20, lr_x=0.1):
        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                assert batch_x.size(0) == batch_size
                loss = self.train_step(batch_x, batch_y, n_iter, lr_x)
                epoch_loss += loss
            print(f"Epoch {epoch}, Energy: {epoch_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    model = BiPCN([2, 4, 2])  # 2->4->2
    x = torch.tensor([[1., 1.],[2., 3.],[4., 5.],[3., 1.],[2., 5.]])
    w_true = torch.tensor([[2., 0.], [0., 2.]])
    y = x @ w_true
    model.fit([(x, y)], batch_size=x.size(0), n_epochs=50, n_iter=20)
    x_list = model.inference(x, n_iter=20)
    print("input:", x)
    print("prediction:", x_list[-1])
    print("target:", y)
