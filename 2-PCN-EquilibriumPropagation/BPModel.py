import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# BP 模型 (普通前馈网络 + BP)
# --------------------------
class BPModel(nn.Module):
    def __init__(self, dims, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1], bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def compute_grads(self, x, y):
        """返回 BP 梯度"""
        out = self.forward(x)
        loss = F.mse_loss(out, y)
        self.zero_grad()
        loss.backward()
        grads = [p.grad.clone() for p in self.parameters()]
        return grads, loss.item()

    def update(self, grads, lr):
        """手动更新权重"""
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                p -= lr * g


# --------------------------
# 测试运行
# --------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(10, 2)
    W_true = torch.tensor([[2., 0.], [0., 2.]])
    y = x @ W_true
    dataloader = [(x, y)]
    epochs = 5

    bp_model = BPModel([2, 2, 2], seed=42)


    for epoch in range(epochs):
        # 外部负责取 batch 和更新模型
        for x_batch, y_batch in dataloader:
            # 更新 BP 模型
            bp_grads, bp_loss = bp_model.compute_grads(x_batch, y_batch)
            bp_model.update(bp_grads, lr=0.01)
            print("epoch", epoch, "bp_loss:", bp_loss)
