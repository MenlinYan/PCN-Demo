import torch
import torch.nn as nn
import torch.nn.functional as F

from BPModel import BPModel
from EPModel import EPModel


# --------------------------
# 梯度对比器
# --------------------------
"""
用于比较 BP 和 EP 模型的梯度相似性
(通过计算每层梯度的余弦相似度)
外部主函数负责初始化 BP 和 EP 模型, 并初始化 GradComparator 梯度对比器
对比器的实现逻辑：
1. 对同一批输入数据, 在训练过程中一边迭代一边调用 compare 方法, 观察梯度相似度的变化
2. 对同一批输入数据, 分别计算 BP 和 EP 模型的各层梯度
3. 对每一层的梯度, 计算 BP 和 EP 梯度的余弦相似度

对比器作为通用比较工具, 对于模型对象的要求：
- 模型对象必须支持自定义初始化网络结构
- 模型对象必须支持自定义随机种子, 用于控制随机初始化参数
- 模型对象必须支持单步梯度计算、更新、读取

自主思考：
- 【问】这里的梯度相似度是否可以作为 EP 模型训练的一个辅助目标？
- 【答】不可以, 这让EP永远不能超过BP, 失去了意义; 但是可以作为一个思路, 如果某两个模型结构导致梯度计算在一定程度上总是关于正确梯度具有一定对称性, 那么可以叠加计算以提高梯度方向准确性
- 【【灵感】】如果进行类似的深入模型内部对比, 就需要模型对象本身支持更细粒度的操作, 把推理/训练过程控制权交给外部
- 【【灵感】】这种对比类实验的通用步骤：
    - 先独立完成每个模型对象, 进行模型内部整体性测试, 确保模型本身没有问题
    - 修改每个模型对象, 开放外部端口, 编写外部主函数进行初始化以及调用测试, 确保模型端口没有问题
    - 编写对比器, 仿照模型对象的主函数写法, 进行对比测试, 确保后续问题的排查可以定位到对比器本身
- 【【提醒】】最后整合成示例代码集合的时候, 要统一的部分：
    - 命名方式：包括文件名、类名、方法名、变量名
    - 能量函数实现
    - 状态形式(parameter/tensor)
    - 梯度计算方式(autograd/手动)
    - 权重更新方式(optimizer/手动)
    - 状态更新方式(optimizer/手动)
    - 初始化方式(随机/前向初始化)
    - Training phase前半部分采用哪种方式收敛(前向初始化/Prediction phase)
- 【问】prediction phase的必要性?为什么不直接用前向初始化替代?
- 【答】=============================================
- ·



"""
class GradComparator:
    """
    通用梯度对比器
    要求模型对象实现:
      - compute_grads(x, y): 返回 (grads, loss)
        其中 grads 为 list[Tensor] (每层权重的梯度)
             loss 为 float (损失值)
    """

    def __init__(self, bp_model, ep_model):
        self.bp_model = bp_model
        self.ep_model = ep_model

    def compare(self, x, y):
        """
        对同一批输入数据, 比较 BP 与 EP 模型的梯度方向相似性
        返回字典:
          {
            "bp_loss": float,
            "ep_loss": float,
            "similarities": [cosine similarity per layer]
          }
        """
        # BP 梯度
        bp_grads, bp_loss = self.bp_model.compute_grads(x, y)
        # EP 梯度
        ep_grads, ep_loss = self.ep_model.compute_grads(x, y)

        similarities = []
        for g_bp, g_ep in zip(bp_grads, ep_grads):
            sim = torch.nn.functional.cosine_similarity(
                g_bp.flatten(), g_ep.flatten(), dim=0
            ).item()
            similarities.append(sim)

        return {
            "bp_loss": bp_loss,
            "ep_loss": ep_loss,
            "similarities": similarities
        }

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

    bp_model = BPModel([2, 4, 2], seed=42)
    ep_model = EPModel([2, 4, 2], beta=1e-3, n_iter=30, seed=42)

    comparator = GradComparator(bp_model, ep_model)

    for epoch in range(epochs):
        # 外部负责取 batch 和更新模型
        for x_batch, y_batch in dataloader:
            # 更新 BP 模型
            bp_grads, bp_loss = bp_model.compute_grads(x_batch, y_batch)
            bp_model.update(bp_grads, lr=0.01)

            # 更新 EP 模型
            ep_grads, ep_loss = ep_model.compute_grads(x_batch, y_batch)
            ep_model.update(ep_grads, lr=0.01)

            # 对比梯度相似性
            # result = comparator.compare(x_batch, y_batch)
            similarities = []
            for g_bp, g_ep in zip(bp_grads, ep_grads):
                sim = torch.nn.functional.cosine_similarity(
                    g_bp.flatten(), g_ep.flatten(), dim=0
                ).item()
                similarities.append(sim)
            result = {
                "bp_loss": bp_loss,
                "ep_loss": ep_loss,
                "similarities": similarities
            }
            print("epoch", epoch, "similarities:", result["similarities"])


"""
    后记：
    - 2025-09-27 12:36:59 可以跑通, GPT的代码实力我是认可的, 但是结果并不好, 吃完饭睡个午觉, 下午继续排查原因
"""