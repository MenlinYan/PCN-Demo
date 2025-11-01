import torch
import torch.nn as nn
import torch.nn.functional as F

# from Tools.PCN_Bi_VIsual import PCN_Bi_Visual
# from Tools.PCN_Visual import PCN_Visual


class BiPCN(nn.Module):
    def __init__(self, dims, device="cpu", seed=42):
        super(BiPCN, self).__init__()
        self.device = device
        self.seed = seed
        self.dims = dims
        self.n_layers = len(dims)

        torch.manual_seed(seed)
        # 前向 (bottom-up) 权重 W: x_{l} -> x_{l+1}
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(d_in, d_out))
            for d_in, d_out in zip(self.dims[:-1], self.dims[1:])
        ])

        # 后向 (top-down) 权重 V: x_{l+1} -> x_{l}
        self.V = nn.ParameterList([
            nn.Parameter(torch.randn(d_out, d_in))
            for d_in, d_out in zip(self.dims[:-1], self.dims[1:])
        ])

        self.optim_W = torch.optim.Adam(
            list(self.W), lr=0.01
        )
        self.optim_V = torch.optim.Adam(
            list(self.V), lr=0.01
        )
        # self.visual_fig = PCN_Visual(depth=self.n_layers, device=device, fixed_xlim=(-5, 5), fixed_ylim=(-5, 5))

    def forward(self, x_list):
        vf_list = [x_list[0]]
        for l in range(len(self.W)):
            vf_list.append(x_list[l] @ self.W[l])
        # vf_list[-1].retain_grad()
        return vf_list

    def backward(self, x_list):
        vb_list = [x_list[-1]]
        for l in range(len(self.V) - 1, -1, -1):
            vb_list.insert(0, x_list[l + 1] @ self.V[l])
        # vb_list[0].retain_grad()
        return vb_list

    def compute_energy(self, x_list, v_list) -> torch.Tensor:
        energy = 0
        # 保证状态与预测状态层数一致
        assert len(x_list) == len(v_list)
        # 遍历层间连接
        for l in range(len(x_list)):
            energy += F.mse_loss(v_list[l], x_list[l], reduction='mean')
        return energy

    def inference(self, x, y=None, n_iter=5, lr=0.1, visualize=False, log=True):
        x_list = [x]
        phase = "Training" if y is None else "Prediction"
        # [DIFFERENCE]
        for i in range(self.n_layers - 1):
            x_next = x_list[-1] @ self.W[i]
            x_list.append(x_next.clone().detach().requires_grad_(True))
        # auto grad x
        begin = 1
        end = self.n_layers if y is None else self.n_layers - 1
        # fix y
        if not y is None:
            x_list[-1].requires_grad = False
            x_list[-1] = y.detach()
        optim_x = torch.optim.SGD(x_list[begin:end], lr=lr)
        for it in range(n_iter):
            v_list = self.forward(x_list)
            E = self.compute_energy(x_list, v_list)
            optim_x.zero_grad()
            E.backward()
            optim_x.step()
            if log and (it % 10 == 0 or it == n_iter - 1):
                print(f'Phase-{phase} "inferring" Iter {it}, Energy: {E.item()}')
            if visualize:
                self.visual_fig.set_x([xi[0].detach().cpu() for xi in x_list])
                self.visual_fig.set_v([vi[0].detach().cpu() for vi in v_list])
                self.visual_fig.visualize(phase="training infer" if y is not None else "prediction infer", rounds=it)
        return x_list, v_list

    def forward_train_step(self, x, y, n_iter=50, lr_x=0.1, lr_w=0.001):
        # 方案1：问题在于v_list在inference中E.backward()之后被清除了，导致train_step中E.backward()报错，需要重新计算一个v_list
        # x_list, v_list = self.inference(x=x, y=y, n_iter=n_iter, lr=0.5, visualize=False)
        # 方案2：像inference中那样重新计算v_list
        x_list, _ = self.inference(x, y, n_iter, lr=lr_x, visualize=False, log=False)
        v_list = self.forward(x_list)
        x_list[-1] = y
        E = self.compute_energy(x_list, v_list)
        self.optim_W.zero_grad()
        E.backward()
        self.optim_W.step()
        return E.item()

    def backward_train_step(self, x, y, n_iter=50, lr_x=0.1, lr_w=0.001):
        x_list, _ = self.inference(x, y, n_iter, lr=lr_x, visualize=False)
        x_list[-1] = y
        v_list = self.backward(x_list)
        E = self.compute_energy(x_list, v_list)
        self.optim_V.zero_grad()
        E.backward()
        self.optim_V.step()
        return E.item()

    def fit(self, train_loader, batch_size=32, n_epochs=10, n_iter=20, lr_x=0.1):
        print("Forward Training")
        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                assert batch_x.size(0) == batch_size
                loss = self.forward_train_step(batch_x, batch_y, n_iter, lr_x)
                epoch_loss += loss
            # print(f"Epoch {epoch}, Energy: {epoch_loss / len(train_loader):.4f}")

        print("Backward Training")
        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                assert batch_x.size(0) == batch_size
                loss = self.backward_train_step(batch_x, batch_y, n_iter, lr_x)
                epoch_loss += loss
            print(f"Epoch {epoch}, Energy: {epoch_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    model = BiPCN([2, 2, 2], seed=42)  # 2->4->2
    x = torch.tensor([[1., 1.], [2., 3.], [4., 5.], [3., 1.], [2., 5.]])
    w_true = torch.tensor([[2., 0.], [0., 2.]])
    y = x @ w_true
    model.fit([(x, y)], batch_size=x.size(0), n_epochs=200, n_iter=100)
    x_list, v_list = model.inference(x, n_iter=20)

    output = y
    for v in reversed(model.V):
        # x_list[l + 1] @ self.V[l]
        output = output @ v

    print("forward prediction:", x_list[-1])
    print("target:", y)
    print("backward prediction:", output)
    print("input:", x)
