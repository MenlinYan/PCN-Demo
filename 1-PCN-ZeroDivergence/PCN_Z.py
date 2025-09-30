import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from Tools.PCN_Visual import PCN_Visual

torch.autograd.set_detect_anomaly(True)
class PCN_Z(nn.Module):
    def __init__(self, dims, device="cpu", B=1, seed=0):
        super(PCN_Z, self).__init__()
        torch.manual_seed(seed)
        self.dims = dims
        self.n_layers = len(dims)
        self.device = device
        self.batch_size = B
        
        # 初始化权重
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(d_in, d_out))
            for d_in, d_out in zip(self.dims[:-1], self.dims[1:])
        ])
        self.states = nn.ParameterList([
            nn.Parameter(torch.randn(self.batch_size, d)) for d in self.dims
        ])

        self.optim_weights = torch.optim.Adam(self.weights, lr=0.01)
        self.visual_fig = PCN_Visual(depth=self.n_layers, device=device, fixed_xlim=(-5, 5), fixed_ylim=(-5, 5))

    def inference(self, x, y=None, n_iter=5, lr=0.1, visualize=False):
        x_list = [x]
        for w in self.weights:
            x_list.append(x_list[-1] @ w)
        return x_list, x_list

    def compute_energy(self, x_list, v_list):
        energy = 0
        for i in range(self.n_layers - 1):
            energy += F.mse_loss(v_list[i+1], x_list[i+1], reduction='mean')
        return energy

    def train_step(self, x, y, lr_w=0.1, visualize=False):
        """_summary_

        Args:
            x (Tensor): (B, dim[0])
            y (Tensor): (B, dim[-1])
            lr_w (float, optional): learning rate of weights. Defaults to 0.01.

        Returns:
            E (float): Energy after training step
        """
        # Step1: 前向初始化 (C1)
        self.states[0] = x.clone()
        for i, w in enumerate(self.weights):
            self.states[i+1] = (self.states[i] @ w).clone().detach()
        self.states[-1] = y.clone().detach()

        # Step2: 
        optimList_x = [torch.optim.SGD([x], lr=1) for x in self.states[1:-1]]
        optimList_w = [torch.optim.Adam([w], lr=lr_w) for w in self.weights]

        # Step3: 逐层更新
        for i in reversed(range(self.n_layers - 2)): 
            # 更新隐含层
            optimList_x[i].zero_grad()
            optimList_w[i].zero_grad()
            v_list = [self.states[0]]
            for j in range(len(self.weights)):
                v_list.append(v_list[j] @ self.weights[j])
            E = self.compute_energy(self.states, v_list)
            E.backward(retain_graph=True)
            optimList_x[i].step()            
            optimList_w[i].step()
            if visualize:
                self.visual_fig.set_x([xi[0].detach().cpu() for xi in self.states])
                self.visual_fig.set_v([vi[0].detach().cpu() for vi in v_list])
                self.visual_fig.visualize(phase="training infer" if y is not None else "prediction infer", rounds=i)
        # 更新w0
        v_list = [self.states[0]]
        for j in range(len(self.weights)):
            v_list.append(v_list[j] @ self.weights[j])
        E = self.compute_energy(self.states, v_list)
        E.backward(retain_graph=True)
        optimList_w[0].zero_grad()
        optimList_w[0].step()
        
        return E.item()


    def fit(self, x, y, batch_size=32, n_epochs=10, n_iter=50, lr_x=0.1, lr_w=0.001, visualize=False):
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(n_epochs):
            epoch_loss = 0  
            for batch_x, batch_y in dataloader:
                epoch_loss += self.train_step(batch_x, batch_y, lr_w=lr_w, visualize=visualize)
            print(f'Epoch {epoch}, Energy: {epoch_loss / len(dataloader)}')

if __name__ == "__main__":
    model = PCN_Z([2, 2, 2], seed=42)
    x = torch.randn(64, 2)
    w = torch.tensor([[2., 0.], [0., 2.]])
    y = x @ w
    B = 32
    model.fit(x, y, batch_size=B, n_epochs=10, lr_w=0.01, visualize=False)

    x_test = torch.tensor([[1., 1.],[2., 3.],[4., 5.],[3., 1.],[2., 5.]])
    y_test = x_test @ w
    x_list, v_list = model.inference(x_test)
    print(x_test)
    print(x_list[-1])
    print(y_test)
