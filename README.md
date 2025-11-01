# PCN-Demo

一些用于学习 **Predictive Coding Networks (PCN)** 的代码示例



##  简介

本仓库提供了一个用于理解和实验预测编码网络（Predictive Coding Networks, PCN）思想的 Python 编码 Demo。重点在于通过几个模块／版本演示 PCN 的不同思想变体（例如“零发散”、平衡传播、双向结构等），便于学习和探索研究方向。



##  仓库结构

```
├── 0-PCN-base/                    # PCN 的基础版本
├── 1-PCN-ZeroDivergence/          # 零发散（Zero Divergence）变体
├── 2-PCN-EquilibriumPropagation/  # 平衡传播 (Equilibrium Propagation) 版
├── 3-PCN-Bidirectional/           # 双向 (Bidirectional) 结构版
├── Tools/                         # 辅助工具脚本（数据处理、可视化等）
├── requirements.txt               # 依赖文件（新增）
├── README.md                      # 本说明文件
└── git_tutorial.md                # Git 教程/参考资料
```


##  快速上手

###  环境要求

* Python ≥ 3.9
* 推荐使用虚拟环境（如 venv 或 conda）进行依赖隔离。

###  环境配置

1. 克隆仓库并安装依赖
    ```bash
    git clone https://github.com/MenlinYan/PCN-Demo.git
    cd PCN-Demo
    pip install -r requirements.txt
    ```

### 模型使用
1. 进入任意子模块（如 `0-PCN-base`），使用IDE直接运行对应模块
2. 查看输出结果（损失曲线、预测误差变化、模型参数更新等）


## 🧠 各子模块简介

| 模块目录                             | 功能简述                            |
| -------------------------------- | ------------------------------- |
| **0-PCN-base**                   | 标准 PCN 基础结构，定义前馈与反馈误差机制         |
| **1-PCN-ZeroDivergence**         | 实现 Zero-Divergence 机制，改善误差信号稳定性 |
| **2-PCN-EquilibriumPropagation** | 基于 Equilibrium Propagation 的版本  |
| **3-PCN-Bidirectional**          | 双向网络结构，信息在高低层间双向流动              |
| **Tools/**                       | 可视化与工具脚本                        |



##  贡献方式

欢迎提交 Issue 或 Pull Request：

* 修复 bug
* 改进结构
* 添加文档或新变体
* 分享实验结果

如果你在使用该仓库代码过程中碰到问题，可以提交issue或直接与我联系
* 邮箱：menlinyan@gmail.com

