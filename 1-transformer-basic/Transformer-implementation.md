# Transformer核心模块详解

## 前言

本文档通过ViT项目解析经典Transformer架构，并对比最新LLM（如DeepSeek V3）中的架构演进。详细解析Transformer的核心模块：位置编码、多头注意力、MLP、LayerNorm和分类头，以及现代LLM中的改进：RoPE、SwiGLU、RMSNorm、MoE等。

---

## 经典Transformer架构 (ViT实现)

以下是基于ViT项目的经典Transformer实现，这是理解现代LLM架构的基础。ViT将Transformer架构成功应用于计算机视觉任务，为后续的多模态大模型奠定了基础。

### 整体架构图

![transformer](https://towardsdatascience.com/wp-content/uploads/2021/01/147UCxMjpfJ2yo48fctNv-g.png)

### 完整ViT实现

```python
class ViT(nn.Module):

    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        num_transformer_layers: int = 12,
        embedding_dim: int = 768,
        mlp_size: int = 3072,
        num_heads: int = 12,
        attn_dropout: float = 0,
        mlp_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()

        assert image_size % patch_size ==0

        # embedding
        self.num_patches = (image_size * image_size) // (patch_size**2)
        self.class_embedding = nn.Parameter(
            data=torch.randn(1, 1, embedding_dim), requires_grad=True
        )
        self.position_embedding = nn.Parameter(
            data=torch.randn(1, self.num_patches + 1, embedding_dim), requires_grad=True
        )
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim
        )

        # transformer decoder
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_size=patch_size,
                    mlp_dropout=mlp_dropout,
                    attn_drop_out=attn_dropout
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    def forward(self,x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x_patched= self.patch_embedding(x)
        x_patched = torch.cat((class_token, x_patched), dim=1)
        x = self.position_embedding + x_patched
        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        x = self.classifier(x[:,0])
        return x
```

### 关键组件说明

- **Patch Embedding**: 将图像分割成16×16的patch，然后将每个patch线性投影到embedding空间
- **Class Token**: 可学习的特殊token，用于聚合全局信息进行分类
- **Position Embedding**: 为每个patch添加位置信息，使模型能够感知空间顺序
- **Transformer Encoder**: 12层transformer块，每层包含多头注意力和MLP
- **Classification Head**: 基于CLS token的最终分类层

---

## 核心模块详解

### 1. 位置编码 (Positional Embedding)

位置编码的目的是为Transformer模型提供序列中每个位置的位置信息，因为Transformer本身是位置无关的。通过不同频率的正弦余弦函数，为不同维度分配不同的位置感知能力。

#### 源码实现

在ViT项目中，位置嵌入的实现如下：

```python
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

#### 数学公式

$$
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$
$$
PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

#### 工作原理

- **低频维度**: 捕捉长距离的位置关系
- **高频维度**: 捕捉短距离的位置细节
- **唯一性**: 每个位置都有独特的编码表示
- **泛化性**: 可以处理训练时未见过的序列长度

### 2. LayerNorm（层归一化）

LayerNorm对每个样本的所有特征进行归一化，使得特征的均值为0，方差为1。然后通过可学习的参数γ（缩放）和β（偏移）来恢复数据的表示能力。

#### 源码实现

```python
class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）

    对每个样本的特征进行归一化，与BatchNorm不同：
    - BatchNorm: 对batch中的同一个特征进行归一化
    - LayerNorm: 对单个样本的所有特征进行归一化
    """

    def __init__(self, num_features: int, eps: float = 1e-6):
        """
        初始化LayerNorm

        Args:
            num_features: 特征维度
            eps: 防止除零的小常数
        """
        super(LayerNorm, self).__init__()
        # 可学习的缩放参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        # 可学习的偏移参数
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            X: 输入张量，形状: (batch_size, seq_len, num_features)

        Returns:
            归一化后的张量
        """
        # 计算均值和方差，沿着最后一个维度
        mean = X.mean(-1, keepdim=True)
        var = X.var(-1, unbiased=False, keepdim=True)

        # 归一化
        X_norm = (X - mean) / torch.sqrt(var + self.eps)

        # 缩放和偏移
        return self.gamma * X_norm + self.beta
```

#### 数学公式

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中：
- $\mu = \frac{1}{n}\sum_{i=1}^{n}x_i$ (均值)
- $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2$ (方差)
- $\gamma, \beta$ 为可学习参数

#### 优势

- **不依赖batch大小**: 对小batch也能正常工作
- **训练一致性**: 训练和推理时行为一致
- **序列友好**: 特别适合RNN和Transformer等序列模型

### 3. 多头注意力 (Multi-Head Attention)

多头注意力机制是Transformer的核心创新，允许模型同时关注来自不同表示子空间的信息。

#### 源码实现
```python
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制（Multi-Head Attention）

    多头注意力允许模型同时关注来自不同表示子空间的信息。
    每个注意力头学习输入序列的不同方面的表示。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        初始化多头注意力

        Args:
            embed_dim: 输入嵌入维度
            num_heads: 注意力头的数量
            dropout: dropout概率
            bias: 是否使用偏置
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 缩放因子，防止梯度消失

        # Q, K, V的线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        need_weights: bool = True
    ) -> tuple:
        """
        前向传播

        Args:
            query: 查询张量，形状: (batch_size, tgt_len, embed_dim)
            key: 键张量，形状: (batch_size, src_len, embed_dim)
            value: 值张量，形状: (batch_size, src_len, embed_dim)
            key_padding_mask: 键填充掩码
            attn_mask: 注意力掩码
            need_weights: 是否返回注意力权重

        Returns:
            注意力输出和注意力权重（如果need_weights=True）
        """
        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)

        # 线性变换得到Q, K, V
        Q = self.q_proj(query)  # (batch_size, tgt_len, embed_dim)
        K = self.k_proj(key)    # (batch_size, src_len, embed_dim)
        V = self.v_proj(value)  # (batch_size, src_len, embed_dim)

        # 重塑为多头格式
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, src_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, src_len, self.num_heads, self.head_dim)

        # 转置维度以支持批量矩阵乘法
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 计算注意力分数
        # (batch_size, num_heads, tgt_len, head_dim) × (batch_size, num_heads, head_dim, src_len)
        # -> (batch_size, num_heads, tgt_len, src_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 应用掩码（如果提供）
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到V
        # (batch_size, num_heads, tgt_len, src_len) × (batch_size, num_heads, src_len, head_dim)
        # -> (batch_size, num_heads, tgt_len, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # 重塑输出
        # (batch_size, num_heads, tgt_len, head_dim) -> (batch_size, tgt_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, tgt_len, embed_dim)

        # 输出投影
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # 返回平均后的注意力权重
            avg_attn_weights = attn_weights.mean(dim=1)
            return attn_output, avg_attn_weights
        else:
            return attn_output, None
```

#### 工作原理

多头注意力机制的核心思想是将输入投影到多个不同的子空间（多个头），在每个子空间中并行计算注意力，然后将所有头的输出拼接并投影回原始维度。

**核心公式**：
```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
```

**工作流程**：
1. **线性投影**：将输入Q、K、V分别通过线性层投影到多个头
2. **注意力计算**：在每个头中计算缩放点积注意力
3. **拼接输出**：将所有头的输出拼接在一起
4. **最终投影**：通过线性层将拼接结果投影回原始维度

**优势**：
- 能够捕捉不同类型的依赖关系
- 增强模型的表达能力
- 并行计算提高效率
- 每个头可以学习不同的特征表示模式

![MHA](https://towardsdatascience.com/wp-content/uploads/2021/01/1NMneDGsvnXyOFqN6m8uSyA.png)

![MHA2](https://substackcdn.com/image/fetch/$s_!Q6zJ!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc1497d87-2b8c-45eb-b7b9-3a0a8ebe0d3d_1783x747.png)

### 4. MLP（多层感知机）

MLP（多层感知机）在Transformer中作为每个位置的非线性变换模块，为模型提供强大的表达能力。

#### 源码实现

```python
class MLPBlock(nn.Module):
    def __init__(
        self, embedding_dim: int = 768, mlp_size: int = 3072, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),  # 768 → 3072
            nn.GELU(),                                                    # 激活函数
            nn.Dropout(p=dropout),                                        # 正则化
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),  # 3072 → 768
            nn.Dropout(p=dropout)                                         # 正则化
        )

    def forward(self, x):
        x = self.layer_norm(x)  # 先归一化
        x = self.mlp(x)         # MLP变换
        return x
```

#### 结构组成详解

1. **LayerNorm (前置归一化)**: 在MLP之前对输入进行归一化，确保稳定的梯度流动
2. **扩展层**: 将维度从 `embedding_dim` 扩展到 `mlp_size` (通常4倍)
3. **GELU激活函数**: Gaussian Error Linear Unit，相比ReLU更平滑
4. **收缩层**: 将维度从 `mlp_size` 收缩回 `embedding_dim`
5. **Dropout正则化**: 防止过拟合，提高泛化能力

#### 工作流程

```
输入 -> LayerNorm -> Linear(扩展) -> GELU -> Dropout -> Linear(收缩) -> Dropout -> 输出
```

#### 设计理念与优势

- **扩展-收缩架构**: 通过中间层扩展增加模型容量，同时保持输入输出维度一致
- **残差连接友好**: 与Transformer的残差连接完美配合
- **计算平衡**: 扩展倍数(4x)在性能和计算成本间取得良好平衡
- **非线性能力**: GELU激活提供强大的非线性建模能力

### 5. 分类头 (Classification Head)

分类模块是Vision Transformer的最后一环，负责将Transformer编码后的特征映射到最终的类别空间。

#### 源码实现

```python
class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim: int = 768, num_classes: int = 1000):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x[:, 0])  # 只使用CLS token
```

#### 设计特点

1. **CLS Token选择**: 使用第一个token进行分类
2. **LayerNorm**: 确保输入分布稳定
3. **线性映射**: 输出类别logits

#### 工作流程

```
Transformer输出 -> 提取CLS token -> LayerNorm -> Linear分类 -> 输出logits
```

#### 为什么使用CLS Token

- **全局信息聚合**: 通过自注意力学会聚合全局信息
- **简洁高效**: 避免复杂池化操作
- **迁移学习友好**: 可作为通用图像特征

## 现代LLM架构改进 (DeepSeek V3对标)

现代LLM在经典Transformer基础上进行了大量优化，以DeepSeek V3为例，以下是关键架构改进：

![DeepSeek](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*3r11L8Luv_L5DkfoPy9rBQ.png)

### 1. LayerNorm → RMSNorm

#### 经典LayerNorm
$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

#### RMSNorm改进
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{RMS}(x)^2 + \epsilon}} \cdot g
$$

其中：$\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}$

#### 核心改进
- **计算效率提升~30%**: 移除均值计算
- **参数量减少50%**: 无需偏移参数β
- **训练稳定性**: 大模型训练中表现更稳定

### 2. GELU → SwiGLU

#### 经典MLPBlock
```python
self.mlp = nn.Sequential(
    nn.Linear(embedding_dim, mlp_size),
    nn.GELU(),
    nn.Linear(mlp_size, embedding_dim),
)
```

#### SwiGLU实现
```python
class SwiGLUMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # Swish门控
        up = self.w_up(x)
        return self.w_down(gate * up)  # 门控机制
```

#### 门控线性单元家族

| 类型 | 公式 | 特点 |
|------|------|------|
| GLU | $(xW_1) \otimes \sigma(xW_2)$ | 经典门控，Sigmoid激活 |
| ReGLU | $(xW_1) \otimes \text{ReLU}(xW_2)$ | 计算简单，稀疏激活 |
| GeGLU | $(xW_1) \otimes \text{GELU}(xW_2)$ | 平滑激活，概率建模 |
| SwiGLU | $\text{Swish}(xW_1) \otimes (xW_2)$ | **最佳性能**，自适应门控 |

#### SwiGLU公式
$$
\text{SwiGLU}(x, W_1, W_2, W_3) = (\text{Swish}(xW_1) \otimes (xW_2))W_3
$$

其中：$\text{Swish}(x) = x \cdot \sigma(x)$

#### 核心改进
- **门控机制**: 动态控制信息流
- **表达能力增强**: 学习复杂非线性变换
- **梯度流动稳定**: 缓解梯度消失

![SwiGLU](https://picx.zhimg.com/70/v2-80ccaa1eba350ba9de547575853da965_1440w.avis?source=172ae18b&biz_tag=Post)

### 3. 绝对位置编码 → RoPE

#### 经典绝对位置编码
```python
# 可学习参数
self.position_embedding = nn.Parameter(torch.randn(1, max_len, embedding_dim))

# 或固定正弦余弦
pos_emb[:, 0::2] = torch.sin(position * div_term)
pos_emb[:, 1::2] = torch.cos(position * div_term)
```

**问题**：
- 序列长度限制
- 外推能力差
- 缺乏相对位置信息

#### RoPE (Rotary Position Embedding)

**核心思想**：通过几何旋转编码位置，使点积只依赖相对位置。

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)
```

#### 高维RoPE公式

$$
\mathbf{x}_t' = \mathbf{R}_{\Theta,t}^d \mathbf{x}_t =
\begin{pmatrix}
\cos t\theta_1 & -\sin t\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin t\theta_1 &  \cos t\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos t\theta_2 & -\sin t\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin t\theta_2 &  \cos t\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos t\theta_{d/2} & -\sin t\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin t\theta_{d/2} & \cos t\theta_{d/2} \\
\end{pmatrix}
\begin{pmatrix}
x_1 \\ x_2 \\ x_3 \\ x_4 \\ \vdots \\ x_{d-1} \\ x_d
\end{pmatrix}
$$

其中旋转频率：$\theta_i = \frac{1}{10000^{2i/d}}$

#### 核心优势

- **完美相对位置**：点积只依赖相对距离
- **强大外推能力**：支持未见过的序列长度
- **零参数开销**：几何变换，无需额外参数
- **计算高效**：旋转矩阵可预计算

### 4. Dense → MoE (Mixture of Experts)

#### 经典Dense架构
```python
# 所有token通过同一个MLP
class MLPBlock(nn.Module):
    def forward(self, x):
        return self.mlp(x)  # 激活所有参数
```

#### MoE架构

```python
class MoE(nn.Module):
    def __init__(self, num_experts=67, top_k=8):
        super().__init__()
        self.experts = nn.ModuleList([SwiGLUMLP(...) for _ in range(num_experts)])
        self.gate = nn.Linear(embed_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # 路由选择激活的专家
        gate_scores = self.gate(x)
        top_k_scores, selected_experts = torch.topk(gate_scores, self.top_k)

        # 只激活选中的专家，大幅节省计算
        final_output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_mask = (selected_experts == i).any()
            if expert_mask.any():
                expert_input = x[expert_mask]
                expert_output = self.experts[i](expert_input)
                final_output[expert_mask] += expert_output * top_k_scores[expert_mask, i:i+1]

        return final_output
```

![Ds MOE](https://assets.zilliz.com/Mo_E_in_Deep_Seek_V3_ee6b538922.png)

#### 核心改进
- **模型稀疏化**: 由Dense转稀疏，节省成本
- **参数规模**: 支持千亿参数，计算成本可控
- **专家并行**: 支持分布式训练

### 5. Multi-Head Attention → MLA (Multi-Head Latent Attention)

#### 经典MHA
```python
class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
```

#### MLA优化
```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, num_heads, latent_dim):
        super().__init__()
        # 低维潜在空间注意力
        self.latent_proj = nn.Linear(embedding_dim, latent_dim)
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, embedding_dim)

    def forward(self, x):
        # 压缩到潜在空间 -> 注意力计算 -> 恢复原始维度
        latent = self.latent_proj(x)
        # 在低维空间计算注意力，减少计算量
```

![MLA](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*tEzj6GIBEW0LEu40nCWdxQ.png)

#### 核心改进
- **计算压缩**: 注意力计算压缩到低维潜在空间
- **KV缓存减少**: 大幅减少内存占用
- **长上下文支持**: 支持更长上下文序列
