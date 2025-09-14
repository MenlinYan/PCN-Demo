import torch
import torch.nn as nn
import torch.nn.functional as F
from PCN_Visual import PCN_Visual


class PCN_Z(nn.Module):
    def __init__(self, dims, device="cpu"):
        super(PCN_Z, self).__init__()
        self.dims = dims
        self.n_layers = len(dims)
        
        # 初始化权重
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(d_in, d_out))
            for d_in, d_out in zip(self.dims[:-1], self.dims[1:])
        ])

        self.optim_weights = torch.optim.Adam(self.weights, lr=0.01)
        self.visual_fig = PCN_Visual(depth=self.n_layers, device=device, fixed_xlim=(-5, 5), fixed_ylim=(-5, 5))

    def compute_energy(self, x_list, v_list):
        energy = 0
        for i in range(self.n_layers - 1):
            energy += F.mse_loss(v_list[i+1], x_list[i+1], reduction='mean')
        return energy

    def forward(self, x_list):
        v_list = [x_list[0]]
        for i in range(len(self.weights)):
            v_list.append(x_list[i] @ self.weights[i])
        v_list[-1].retain_grad()  # 保留最后一层的梯度
        return v_list

    def inference(self, x, y=None, n_iter=5, lr=0.1, visualize=False):
        x_list = [x]
        
        # 逐层推理
        for i in range(self.n_layers - 1):
            x_next = x_list[-1] @ self.weights[i]
            x_list.append(x_next.clone().detach().requires_grad_(True))

        # 初始化优化器（对 x 进行优化）
        begin = 1
        end = self.n_layers if y is None else self.n_layers - 1
        if y is not None:
            x_list[-1].requires_grad = False
            x_list[-1] = y.detach()

        optim_x = torch.optim.SGD(x_list[begin:end], lr=lr)
        
        # Z-PCN推理
        for it in range(n_iter):
            v_list = self.forward(x_list)
            E = self.compute_energy(x_list, v_list)
            optim_x.zero_grad()
            E.backward()  # 反向传播
            optim_x.step()  # 更新

            # 打印训练信息
            if it % 10 == 0 or it == n_iter - 1:
                print(f'Phase-{"training infer" if y is not None else "prediction infer"} Iter {it}, Energy: {E.item()}')
            
            # 可视化
            if visualize:
                self.visual_fig.set_x([xi[0].detach().cpu() for xi in x_list])
                self.visual_fig.set_v([vi[0].detach().cpu() for vi in v_list])
                self.visual_fig.visualize(phase="training infer" if y is not None else "prediction infer", rounds=it)

        return x_list, v_list

    def train_step(self, x, y, n_iter=50, lr_x=0.1, lr_w=0.001):
        x_list, _ = self.inference(x, y, n_iter, lr=lr_x, visualize=False)
        v_list = self.forward(x_list)
        
        # 更新最后一层输出为目标y
        x_list[-1] = y
        E = self.compute_energy(x_list, v_list)
        
        self.optim_weights.zero_grad()
        E.backward()  # 更新权重
        self.optim_weights.step()
        return E.item()

    def fit(self, train_loader, batch_size=32, n_epochs=10, n_iter=50, lr_x=0.1, lr_w=0.001):
        for epoch in range(n_epochs):
            epoch_loss = 0  
            for n_batch, (batch_x, batch_y) in enumerate(train_loader):
                assert batch_x.size(0) == batch_size
                epoch_loss += self.train_step(batch_x, batch_y, n_iter, lr_x, lr_w)
            print(f'Epoch {epoch}, Energy: {epoch_loss / len(train_loader)}')

if __name__ == "__main__":
    model = PCN_Z([2, 2, 2])
    x = torch.tensor([[1., 1.],[2., 3.],[4., 5.],[3., 1.],[2., 5.]])
    B = x.size(0)
    w = torch.tensor([[2., 0.], [0., 2.]])
    y = x @ w
    model.fit([(x, y)], batch_size=B, n_epochs=200, n_iter=30)
    x_list, v_list = model.inference(x, n_iter=20)
    print(x)
    print(x_list[-1])
    print(y)
