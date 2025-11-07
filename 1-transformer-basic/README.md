# ViT项目 - Transformer实战

Vision Transformer实战项目，帮你理解Transformer怎么工作的。

## 项目内容

### 📖 理论文档
**Transformer-implementation.md** - 详细解释Transformer的五个核心模块：
- 位置编码、多头注意力、MLP、LayerNorm、分类头
- 还有DeepSeek V3等现代LLM的架构改进（RoPE、SwiGLU、RMSNorm、MoE等）
- 包含数学公式和代码实现

### 💻 实战代码
**vit/** 目录下有三个jupyter notebook：

1. **`train.ipynb`** - 从零开始训ViT
   - 完整实现，150个epoch
   - 需要GPU（RTX 3070可跑）
   - 适合深入理解原理

2. **`train_transfer_learning.ipynb`** - 迁移学习 ⭐推荐
   - 用预训练模型，只训分类头
   - 30个epoch就有不错效果
   - 训练快，适合入门

3. **`prediction.ipynb`** - 模型测试
   - 混淆矩阵、性能指标
   - 结果可视化

建议先跑 `train_transfer_learning.ipynb`，快速看到效果！

## 数据集

公开的小数据集（pizza/steak/sushi三分类），首次运行时自动下载。


>项目地址：[Nicooo-Wang/LLM-inference-basics](https://github.com/Nicooo-Wang/LLM-inference-basics/tree/main)