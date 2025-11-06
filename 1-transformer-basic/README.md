# ViT项目 - Transformer实战

这里是一个Vision Transformer的实战项目，用来理解Transformer是怎么工作的。

## 三个notebook干什么的

1. **`train.ipynb`** - 从零开始训一个ViT模型
   - 完整实现，150个epoch，训练比较慢，需要一定显存，实测3070能跑起来，npu没试过
   - 适合想深入理解原理的同学

2. **`train_transfer_learning.ipynb`** - 迁移学习，训练快！
   - 用预训练模型，只训分类头
   - 30个epoch就有不错效果
   - **推荐从这里开始**

3. **`prediction.ipynb`** - 模型测试和可视化
   - 加载训好的模型做预测
   - 画混淆矩阵，算性能指标

## Transformer-implementation.md

技术文档，详细解释了Transformer的五个核心模块：
- 位置编码、多头注意力、MLP、LayerNorm、分类头
- 有代码和数学公式，想深挖原理的时候看

## 数据集

用的是github上一个公开的小数据集，自动下载。

## 快速开始

```bash
cd 1-transformer-basic/vit
uv venv && source .venv/bin/activate
uv sync
jupyter lab
```

建议先跑 `train_transfer_learning.ipynb`，快速看到效果！