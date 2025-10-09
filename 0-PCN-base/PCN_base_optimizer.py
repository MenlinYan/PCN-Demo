import torch
import torch.nn as nn
import torch.nn.functional as F
from Tools.PCN_Visual import PCN_Visual

class PCNBaseOptimizer(nn.Module):
    def __init__(self, dims, device="cpu", seed=42):
        super(PCNBaseOptimizer, self).__init__()
        self.dims = dims
        # self.n_layers = len(dims) - 1
        torch.manual_seed(seed)
        self.n_layers = len(dims)
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
        # TODO：不知道为什么，加上下面这句效果会好一些
        v_list[-1].retain_grad()
        return v_list
    
    def inference(self, x, y=None, n_iter=5, lr=0.1, visualize=False):
        x_list = [x]
        # [DIFFERENCE]
        for i in range(self.n_layers - 1):
            x_next = x_list[-1] @ self.weights[i]
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
            if it % 10 == 0 or it == n_iter - 1:
                print(f'Phase-{"training infer" if y is not None else "prediction infer"} Iter {it}, Energy: {E.item()}')
            if visualize:
                self.visual_fig.set_x([xi[0].detach().cpu() for xi in x_list])
                self.visual_fig.set_v([vi[0].detach().cpu() for vi in v_list])
                self.visual_fig.visualize(phase="training infer" if y is not None else "prediction infer", rounds=it)
        return x_list, v_list

    def train_step(self, x, y, n_iter=50, lr_x=0.1, lr_w=0.001):
        # 方案1：问题在于v_list在inference中E.backward()之后被清除了，导致train_step中E.backward()报错，需要重新计算一个v_list
        # x_list, v_list = self.inference(x=x, y=y, n_iter=n_iter, lr=0.5, visualize=False)
        # 方案2：像inference中那样重新计算v_list
        x_list, _ = self.inference(x, y, n_iter, lr=lr_x, visualize=False)
        v_list = self.forward(x_list)
        x_list[-1] = y
        E = self.compute_energy(x_list, v_list)
        self.optim_weights.zero_grad()
        E.backward()
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
    model = PCNBaseOptimizer([2, 2, 2], seed=42)
    x = torch.tensor([[1., 1.],[2., 3.],[4., 5.],[3., 1.],[2., 5.]])
    B = x.size(0)
    w = torch.tensor([[2., 0.], [0., 2.]])
    y = x @ w
    model.fit([(x, y)], batch_size=B, n_epochs=200, n_iter=30)
    x_list, v_list = model.inference(x, n_iter=20)
    print(x)
    print(x_list[-1])
    print(y)
    