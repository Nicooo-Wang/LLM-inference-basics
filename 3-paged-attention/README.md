# PagedAttention 与内存管理

---

## 一、开篇引入

### 1.1 KV Cache 的内存瓶颈

在 Transformer 架构的大语言模型推理过程中，每生成一个新 token 都需要用到之前所有 token 的 Key 和 Value 向量。为避免重复计算，我们会将这些 KV 向量缓存起来，这就是 **KV Cache**。

KV Cache 的显存占用可以用以下公式估算：

```
KV Cache Size = 2 × num_layers × seq_len × num_kv_heads × head_dim × dtype_size
```

以 Qwen3-0.6B 为例（28层，8个KV头，64维，bf16）：
- 单个请求、序列长度 4096：`2 × 28 × 4096 × 8 × 64 × 2 = 234 MB`
- 如果同时服务 32 个请求：约 **7.34 GB** 显存仅用于 KV Cache

传统方案的问题在于 **预分配最大长度**。假设 `max_model_len = 4096`，即使一个请求实际只生成 100 个 token，也会预分配 4096 长度的 KV Cache 空间，导致 **60%-80% 的显存浪费**。此外，不同请求的实际长度参差不齐，容易产生 **显存碎片**。
![pic](https://cdn.prod.website-files.com/618399cd49d125734c8dec95/663e10712d1f683554c11f56_UubwS2ISHuve8WjvuB6QaIzh9MZzmLNC5Rz4EhJ03CS_6u9KrhRMkWHexslnmdTYCCjEp07aqg3sBmQy-63FxZnSNUTYOr1V-7Dr53qq4PvpERJaxf3DAKU_lKIFSHxSl7BbZpZDKruOOYkNDft7YaQ.png)
### 1.2 PagedAttention 核心思想

PagedAttention 借鉴了操作系统虚拟内存管理的 **分页机制**：

| 操作系统概念 | PagedAttention 类比 |
|-------------|-------------------|
| 页（Page） | Block（固定大小的 KV Cache 块） |
| 页表（Page Table） | Block Table（逻辑位置到物理 Block 的映射） |
| 物理页框（Frame） | 预分配的 GPU 显存中的 Block 槽位 |
| 进程（Process） | Sequence（一个推理请求） |

核心优势：
- **按需分配**：只为实际生成的 token 分配 Block
- **动态管理**：请求结束后立即回收 Block 供其他请求使用
- **支持共享**：相同前缀的请求可共享 Block（Prefix Caching）

---

## 二、nano-vllm 代码结构总览

### 2.1 相关文件定位

```
nanovllm/
├── engine/
│   ├── block_manager.py   # Block 分配与回收、Prefix Caching
│   ├── sequence.py        # Sequence 数据结构（含 block_table）
│   ├── scheduler.py       # 调度器（调用 BlockManager）
│   └── model_runner.py    # KV Cache 物理显存分配、上下文准备
├── layers/
│   └── attention.py       # Attention 计算 + KV Cache 存取
├── utils/
│   └── context.py         # 运行时上下文（slot_mapping、block_tables等）
└── config.py              # kvcache_block_size 等配置
```

### 2.2 模块协作关系图

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .arch-container {
            background-color: white;
            padding: 40px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .arch-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 30px;
            text-align: center;
            color: #333;
        }
        .arch-diagram {
            position: relative;
            width: 750px;
            height: 420px;
            margin: 0 auto;
        }
        .arch-module {
            position: absolute;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 12px 16px;
            text-align: center;
            font-size: 13px;
            font-weight: bold;
        }
        .arch-file {
            font-size: 10px;
            font-weight: normal;
            color: #666;
            margin-top: 4px;
        }
        .arch-scheduler { background: #E3F2FD; left: 20px; top: 160px; width: 100px; }
        .arch-block-mgr { background: #FFF3E0; left: 180px; top: 160px; width: 120px; }
        .arch-sequence { background: #E8F5E9; left: 360px; top: 160px; width: 100px; }
        .arch-model-runner { background: #FCE4EC; left: 520px; top: 80px; width: 120px; }
        .arch-context { background: #F3E5F5; left: 520px; top: 240px; width: 120px; }
        .arch-attention { background: #E0F7FA; left: 520px; top: 160px; width: 120px; }
        .arch-kv-cache { background: #FBE9E7; left: 680px; top: 160px; width: 60px; border-style: dashed; }
        .arch-legend {
            position: absolute;
            bottom: 0;
            left: 0;
            font-size: 11px;
            color: #666;
        }
        .arch-flow {
            position: absolute;
            top: 20px;
            left: 0;
            right: 0;
            text-align: center;
        }
        .arch-flow-box {
            display: inline-block;
            padding: 8px 20px;
            margin: 0 5px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .arch-flow-arrow {
            display: inline-block;
            color: #999;
            font-size: 16px;
        }
        .arch-desc {
            position: absolute;
            font-size: 10px;
            color: #666;
            max-width: 80px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="arch-container">
        <div class="arch-title">nano-vllm PagedAttention 模块协作图</div>
        <div class="arch-diagram">
            <svg width="750" height="350" style="position: absolute; top: 40px; left: 0;">
                <defs>
                    <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
                    </marker>
                </defs>
                <!-- Scheduler -> BlockManager -->
                <line x1="125" y1="180" x2="175" y2="180" stroke="#666" stroke-width="2" marker-end="url(#arrow)"/>
                <!-- BlockManager -> Sequence -->
                <line x1="305" y1="180" x2="355" y2="180" stroke="#666" stroke-width="2" marker-end="url(#arrow)"/>
                <!-- Sequence -> ModelRunner -->
                <path d="M 465 160 Q 490 100 515 100" fill="none" stroke="#666" stroke-width="2" marker-end="url(#arrow)"/>
                <!-- ModelRunner -> Context -->
                <line x1="580" y1="145" x2="580" y2="195" stroke="#666" stroke-width="2" marker-end="url(#arrow)"/>
                <!-- ModelRunner -> Attention -->
                <line x1="580" y1="145" x2="580" y2="155" stroke="#666" stroke-width="2"/>
                <!-- Context -> Attention -->
                <line x1="580" y1="235" x2="580" y2="225" stroke="#666" stroke-width="2" marker-end="url(#arrow)"/>
                <!-- Attention -> KV Cache -->
                <line x1="645" y1="180" x2="675" y2="180" stroke="#666" stroke-width="2" marker-end="url(#arrow)"/>
            </svg>
            
            <div class="arch-flow">
                <span class="arch-flow-box" style="background: #BBDEFB;">1. 调度</span>
                <span class="arch-flow-arrow">→</span>
                <span class="arch-flow-box" style="background: #FFE0B2;">2. 分配Block</span>
                <span class="arch-flow-arrow">→</span>
                <span class="arch-flow-box" style="background: #C8E6C9;">3. 更新映射</span>
                <span class="arch-flow-arrow">→</span>
                <span class="arch-flow-box" style="background: #F8BBD9;">4. 准备上下文</span>
                <span class="arch-flow-arrow">→</span>
                <span class="arch-flow-box" style="background: #B2EBF2;">5. 执行Attention</span>
            </div>
            
            <div class="arch-module arch-scheduler">
                Scheduler
                <div class="arch-file">scheduler.py</div>
            </div>
            <div class="arch-module arch-block-mgr">
                BlockManager
                <div class="arch-file">block_manager.py</div>
            </div>
            <div class="arch-module arch-sequence">
                Sequence
                <div class="arch-file">sequence.py</div>
            </div>
            <div class="arch-module arch-model-runner">
                ModelRunner
                <div class="arch-file">model_runner.py</div>
            </div>
            <div class="arch-module arch-context">
                Context
                <div class="arch-file">context.py</div>
            </div>
            <div class="arch-module arch-attention">
                Attention
                <div class="arch-file">attention.py</div>
            </div>
            <div class="arch-module arch-kv-cache">
                KV<br>Cache
            </div>
            
            <div class="arch-desc" style="left: 125px; top: 210px;">allocate<br>deallocate</div>
            <div class="arch-desc" style="left: 305px; top: 210px;">block_table</div>
            <div class="arch-desc" style="left: 640px; top: 210px;">read/write</div>
            
            <div class="arch-legend">
                * 实线箭头表示主要数据/调用流向，虚线框表示 GPU 显存
            </div>
        </div>
    </div>
</body>
</html>
```

**核心协作流程**：

1. **Scheduler** 决定哪些 Sequence 参与本次迭代
2. **BlockManager** 为 Sequence 分配/回收 Block，更新 Sequence 的 `block_table`
3. **ModelRunner** 读取 Sequence 的 `block_table`，构造 `slot_mapping` 等上下文信息
4. **Context** 保存运行时上下文，供 Attention 层使用
5. **Attention** 根据上下文读写 KV Cache，完成注意力计算

---

## 三、核心类与方法详解

### 3.1 Block 类（block_manager.py）

Block 是 PagedAttention 的最小存储单元，代表 KV Cache 中的一个固定大小的槽位。

```python
class Block:

    def __init__(self, block_id):
        self.block_id = block_id      # 物理 Block 的唯一标识
        self.ref_count = 0            # 引用计数，支持 Prefix Caching 共享
        self.hash = -1                # Block 内容的 hash 值
        self.token_ids = []           # Block 中存储的 token IDs

    def update(self, hash: int, token_ids: list[int]):
        """更新 Block 的缓存标识，用于 Prefix Caching"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置 Block 状态，供新分配使用"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
```

**字段详解**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `block_id` | int | 物理 Block 在 KV Cache Tensor 中的索引，创建后不变 |
| `ref_count` | int | 引用计数。=0 表示空闲，=1 表示被一个 Sequence 使用，>1 表示被多个 Sequence 共享 |
| `hash` | int | Block 内容的 xxhash 值。=-1 表示 Block 未填满或不参与缓存 |
| `token_ids` | list[int] | Block 对应的完整 token 序列，用于 hash 碰撞时的校验 |

**设计意图**：
- `hash` 用于快速查找：O(1) 判断是否存在相同内容的 Block
- `token_ids` 用于精确校验：防止 hash 碰撞导致的错误命中
- 只有 **填满** 的 Block 才会设置 `hash` 和 `token_ids`

### 3.2 BlockManager 类（block_manager.py）

BlockManager 是 Block 的管理器，负责分配、回收和 Prefix Caching。

```python
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
```

**字段详解**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `block_size` | int | 每个 Block 容纳的 token 数，默认 256 |
| `blocks` | list[Block] | 所有物理 Block 实例，索引即 `block_id` |
| `hash_to_block_id` | dict[int, int] | hash → block_id 映射，Prefix Caching 的核心索引 |
| `free_block_ids` | deque[int] | 空闲 Block ID 队列，FIFO 分配 |
| `used_block_ids` | set[int] | 已使用的 Block ID 集合，O(1) 查询 |

**核心方法**：

#### 3.2.1 `compute_hash(token_ids, prefix) -> int`

计算 Block 内容的 hash 值，支持链式传递前缀 hash。

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))  # 包含前缀 hash
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

**链式 hash 的意义**：Block 2 的 hash 依赖于 Block 0 和 Block 1 的内容，确保只有**完全相同的前缀**才能匹配。

#### 3.2.2 `can_allocate(seq) -> bool`

检查是否有足够的空闲 Block 为 Sequence 分配。

```python
def can_allocate(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= seq.num_blocks
```

**注意**：这是保守估计，未考虑 Prefix Caching 命中可能减少的分配需求。

#### 3.2.3 `allocate(seq)`

为 Sequence 分配 Block，核心方法，包含完整的 Prefix Caching 逻辑。

```python
def allocate(self, seq: Sequence):
    assert not seq.block_table  # 确保是新请求
    h = -1                       # 前缀 hash
    cache_miss = False           # 一旦 miss，后续都是 miss
    
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        # 只有完整 Block 才计算 hash
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)
        
        # hash 命中但内容不匹配，视为 miss
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        
        if cache_miss:
            # Cache Miss：从空闲池分配新 Block
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # Cache Hit：复用已有 Block
            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                block = self._allocate_block(block_id)
        
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        
        seq.block_table.append(block_id)
```

#### 3.2.4 `can_append(seq) -> bool`

检查是否能为 Sequence 追加新的 token（可能需要新 Block）。

```python
def can_append(self, seq: Sequence) -> bool:
    # 只有当需要新 Block 时才检查空闲池
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

#### 3.2.5 `may_append(seq)`

Decode 阶段追加 Block，处理三种情况：

```python
def may_append(self, seq: Sequence):
    block_table = seq.block_table
    last_block = self.blocks[block_table[-1]]
    
    if len(seq) % self.block_size == 1:
        # 情况1：需要新 Block（上一个已满）
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        block_table.append(block_id)
        
    elif len(seq) % self.block_size == 0:
        # 情况2：当前 Block 刚填满，更新 hash
        token_ids = seq.block(seq.num_blocks - 1)
        prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
        h = self.compute_hash(token_ids, prefix)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id
        
    else:
        # 情况3：Block 正在填充中，无需操作
        pass
```

#### 3.2.6 `deallocate(seq)`

释放 Sequence 占用的所有 Block。

```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

**使用示例**：

```python
# 典型调用顺序
block_manager = BlockManager(num_blocks=100, block_size=256)

# 新请求到达
if block_manager.can_allocate(seq):
    block_manager.allocate(seq)

# Decode 阶段
if block_manager.can_append(seq):
    block_manager.may_append(seq)

# 请求完成
block_manager.deallocate(seq)
```

### 3.3 Sequence 类中的 Block 相关部分（sequence.py）

Sequence 代表一个推理请求，其中与 Block 相关的字段和方法如下：

```python
class Sequence:
    block_size = 256  # 类变量
    
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.token_ids = copy(token_ids)
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0      # Prefix Cache 命中的 token 数
        self.block_table = []           # 逻辑 Block → 物理 Block ID
        # ...
```

**核心属性**：

```python
@property
def num_blocks(self):
    """当前 Sequence 需要的 Block 总数"""
    return (self.num_tokens + self.block_size - 1) // self.block_size

@property
def num_cached_blocks(self):
    """Prefix Cache 命中的 Block 数"""
    return self.num_cached_tokens // self.block_size

@property
def last_block_num_tokens(self):
    """最后一个 Block 中的 token 数量"""
    return self.num_tokens - (self.num_blocks - 1) * self.block_size

def block(self, i):
    """获取第 i 个逻辑 Block 对应的 token IDs"""
    return self.token_ids[i * self.block_size : (i + 1) * self.block_size]
```

**block_table 映射关系图**：

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .bt-container {
            background-color: white;
            padding: 30px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .bt-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333;
        }
        .bt-section {
            margin-bottom: 25px;
        }
        .bt-label {
            font-weight: bold;
            color: #555;
            margin-bottom: 8px;
        }
        .bt-row {
            display: flex;
            align-items: center;
        }
        .bt-cell {
            width: 70px;
            height: 45px;
            border: 2px solid #333;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 8px;
            font-size: 12px;
            text-align: center;
            border-radius: 4px;
        }
        .bt-logical { background: #E3F2FD; }
        .bt-tokens { background: #E8F5E9; }
        .bt-physical { background: #FFF3E0; }
        .bt-last { border-color: #F44336; border-width: 3px; }
        .bt-arrow-section {
            display: flex;
            justify-content: center;
            margin: 15px 0;
        }
        .bt-arrow-group {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 30px;
        }
        .bt-arrow {
            font-size: 24px;
            color: #666;
        }
        .bt-note {
            font-size: 11px;
            color: #888;
            margin-top: 15px;
        }
        .bt-formula {
            background: #F5F5F5;
            padding: 10px 15px;
            border-radius: 5px;
            font-family: 'Consolas', monospace;
            font-size: 13px;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <div class="bt-container">
        <div class="bt-title">Sequence 的 block_table 映射示例</div>
        <div class="bt-section">
            <div class="bt-label">假设：block_size = 4，Sequence 有 10 个 token</div>
        </div>
        
        <div class="bt-section">
            <div class="bt-label">逻辑 Block 索引：</div>
            <div class="bt-row">
                <div class="bt-cell bt-logical">Block 0</div>
                <div class="bt-cell bt-logical">Block 1</div>
                <div class="bt-cell bt-logical bt-last">Block 2</div>
            </div>
        </div>
        
        <div class="bt-section">
            <div class="bt-label">Token 分布：</div>
            <div class="bt-row">
                <div class="bt-cell bt-tokens">t0,t1,t2,t3</div>
                <div class="bt-cell bt-tokens">t4,t5,t6,t7</div>
                <div class="bt-cell bt-tokens bt-last">t8,t9</div>
            </div>
        </div>
        
        <div class="bt-arrow-section">
            <div class="bt-arrow-group">
                <div class="bt-arrow">↓</div>
                <div style="font-size: 11px; color: #666;">block_table[0]=7</div>
            </div>
            <div class="bt-arrow-group">
                <div class="bt-arrow">↓</div>
                <div style="font-size: 11px; color: #666;">block_table[1]=3</div>
            </div>
            <div class="bt-arrow-group">
                <div class="bt-arrow">↓</div>
                <div style="font-size: 11px; color: #666;">block_table[2]=12</div>
            </div>
        </div>
        
        <div class="bt-section">
            <div class="bt-label">物理 Block ID（在 KV Cache Tensor 中的位置）：</div>
            <div class="bt-row">
                <div class="bt-cell bt-physical">Physical 7</div>
                <div class="bt-cell bt-physical">Physical 3</div>
                <div class="bt-cell bt-physical bt-last">Physical 12</div>
            </div>
        </div>
        
        <div class="bt-formula">
            block_table = [7, 3, 12]<br>
            num_blocks = 3<br>
            num_cached_blocks = 0（假设无缓存命中）<br>
            last_block_num_tokens = 10 - 2×4 = 2
        </div>
        
        <div class="bt-note">* 红框标注的是最后一个 Block，可能未填满</div>
    </div>
</body>
</html>
```

### 3.4 KV Cache 物理存储（model_runner.py）

KV Cache 的物理存储是一个预分配的 GPU Tensor，由 `ModelRunner.allocate_kv_cache()` 方法创建。

**Tensor 形状**：

```
[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
 │      │           │           │            │           │
 │      │           │           │            │           └─ 每个头的维度
 │      │           │           │            └─ KV 头数量
 │      │           │           └─ 每个 Block 的槽位数
 │      │           └─ 物理 Block 总数
 │      └─ Transformer 层数
 └─ 0=Key, 1=Value
```

**分配方法**：

```python
def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config
    
    # 1. 获取显存信息
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    
    # 2. 计算单个 Block 的显存占用
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    block_bytes = (2 * hf_config.num_hidden_layers * self.block_size * 
                   num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize)
    
    # 3. 计算可分配的 Block 数量
    available = total * config.gpu_memory_utilization - used - peak + current
    config.num_kvcache_blocks = int(available) // block_bytes
    
    # 4. 预分配 Tensor
    self.kv_cache = torch.empty(
        2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
        self.block_size, num_kv_heads, hf_config.head_dim
    )
    
    # 5. 绑定到每个 Attention 层
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

**KV Cache Tensor 结构图**：

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .kv-container {
            background-color: white;
            padding: 30px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .kv-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 25px;
        }
        .kv-dim-row {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
        }
        .kv-dim-label {
            width: 220px;
            font-size: 13px;
            color: #333;
        }
        .kv-dim-boxes {
            display: flex;
            align-items: center;
        }
        .kv-box {
            min-width: 45px;
            height: 32px;
            border: 1px solid #666;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            margin-right: 3px;
            padding: 0 5px;
            border-radius: 3px;
        }
        .kv-k { background: #BBDEFB; }
        .kv-v { background: #FFCCBC; }
        .kv-layer { background: #C8E6C9; }
        .kv-block { background: #FFF9C4; }
        .kv-slot { background: #E1BEE7; }
        .kv-head { background: #B2EBF2; }
        .kv-dim { background: #F5F5F5; }
        .kv-dots {
            margin: 0 5px;
            color: #999;
            font-weight: bold;
        }
        .kv-example {
            margin-top: 25px;
            padding: 15px;
            background: #FAFAFA;
            border-left: 4px solid #2196F3;
            font-size: 13px;
        }
        .kv-code {
            font-family: 'Consolas', monospace;
            background: #ECEFF1;
            padding: 2px 6px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="kv-container">
        <div class="kv-title">KV Cache Tensor 维度结构</div>
        
        <div class="kv-dim-row">
            <div class="kv-dim-label"><strong>Dim 0</strong> — K/V 选择器：</div>
            <div class="kv-dim-boxes">
                <div class="kv-box kv-k">0: Key</div>
                <div class="kv-box kv-v">1: Value</div>
            </div>
        </div>
        
        <div class="kv-dim-row">
            <div class="kv-dim-label"><strong>Dim 1</strong> — Transformer 层：</div>
            <div class="kv-dim-boxes">
                <div class="kv-box kv-layer">Layer 0</div>
                <div class="kv-box kv-layer">Layer 1</div>
                <span class="kv-dots">...</span>
                <div class="kv-box kv-layer">Layer 27</div>
            </div>
        </div>
        
        <div class="kv-dim-row">
            <div class="kv-dim-label"><strong>Dim 2</strong> — 物理 Block：</div>
            <div class="kv-dim-boxes">
                <div class="kv-box kv-block">Block 0</div>
                <div class="kv-box kv-block">Block 1</div>
                <span class="kv-dots">...</span>
                <div class="kv-box kv-block">Block N</div>
            </div>
        </div>
        
        <div class="kv-dim-row">
            <div class="kv-dim-label"><strong>Dim 3</strong> — Block 内槽位：</div>
            <div class="kv-dim-boxes">
                <div class="kv-box kv-slot">Slot 0</div>
                <div class="kv-box kv-slot">Slot 1</div>
                <span class="kv-dots">...</span>
                <div class="kv-box kv-slot">Slot 255</div>
            </div>
        </div>
        
        <div class="kv-dim-row">
            <div class="kv-dim-label"><strong>Dim 4</strong> — KV 头：</div>
            <div class="kv-dim-boxes">
                <div class="kv-box kv-head">Head 0</div>
                <div class="kv-box kv-head">Head 1</div>
                <span class="kv-dots">...</span>
                <div class="kv-box kv-head">Head 7</div>
            </div>
        </div>
        
        <div class="kv-dim-row">
            <div class="kv-dim-label"><strong>Dim 5</strong> — 头维度：</div>
            <div class="kv-dim-boxes">
                <div class="kv-box kv-dim">d₀</div>
                <div class="kv-box kv-dim">d₁</div>
                <span class="kv-dots">...</span>
                <div class="kv-box kv-dim">d₁₂₇</div>
            </div>
        </div>
        
        <div class="kv-example">
            <strong>Qwen3-0.6B 示例：</strong><br><br>
            Shape: <span class="kv-code">[2, 28, N, 256, 8, 128]</span><br><br>
            其中 N = num_kvcache_blocks，由可用显存自动计算<br><br>
            单个 Block 显存：<span class="kv-code">2 × 28 × 256 × 8 × 128 × 2 = 29.36 MB</span><br><br>
            若可用显存 4.6 GB → N ≈ <strong>160</strong> 个 Block
        </div>
    </div>
</body>
</html>
```

### 3.5 Context 上下文（context.py）

Context 是运行时上下文的容器，保存 Attention 计算所需的各种索引信息。

```python
@dataclass
class Context:
    is_prefill: bool = False                    # 当前是否为 Prefill 阶段
    cu_seqlens_q: torch.Tensor | None = None    # Query 累积序列长度
    cu_seqlens_k: torch.Tensor | None = None    # Key 累积序列长度
    max_seqlen_q: int = 0                       # Query 最大序列长度
    max_seqlen_k: int = 0                       # Key 最大序列长度
    slot_mapping: torch.Tensor | None = None    # token → Cache 槽位映射
    context_lens: torch.Tensor | None = None    # 每个序列的上下文长度（Decode）
    block_tables: torch.Tensor | None = None    # 批量 block_table（Decode/Prefix Cache）
```

**字段详解**：

| 字段 | Prefill 阶段 | Decode 阶段 | 说明 |
|------|-------------|-------------|------|
| `is_prefill` | True | False | 区分阶段 |
| `cu_seqlens_q` | ✅ 使用 | ❌ 不使用 | 累积序列长度，如 [0, 100, 250, 400] |
| `cu_seqlens_k` | ✅ 使用 | ❌ 不使用 | 与 cu_seqlens_q 类似，Prefix Cache 时可能不同 |
| `max_seqlen_q` | ✅ 使用 | ❌ 不使用 | 批次中最长的 Query 长度 |
| `max_seqlen_k` | ✅ 使用 | ❌ 不使用 | 批次中最长的 Key 长度 |
| `slot_mapping` | ✅ 使用 | ✅ 使用 | token 在 KV Cache 中的槽位 |
| `context_lens` | ❌ 不使用 | ✅ 使用 | 每个序列的总长度 |
| `block_tables` | Prefix Cache 时使用 | ✅ 使用 | 2D Tensor，每行是一个 block_table |

**设置与获取**：

```python
# 设置上下文
set_context(
    is_prefill=True,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    slot_mapping=slot_mapping,
    block_tables=block_tables
)

# 获取上下文（在 Attention 中调用）
context = get_context()
if context.is_prefill:
    # Prefill 逻辑
else:
    # Decode 逻辑

# 重置上下文
reset_context()
```

### 3.6 Attention 层的 KV Cache 操作（attention.py）

#### 3.6.1 `store_kvcache_kernel`：Triton Kernel

将 K、V 向量写入 KV Cache 的指定槽位。

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,            # 输入 K 的指针
    key_stride,         # K 的步长
    value_ptr,          # 输入 V 的指针
    value_stride,       # V 的步长
    k_cache_ptr,        # K Cache 的指针
    v_cache_ptr,        # V Cache 的指针
    slot_mapping_ptr,   # slot_mapping 的指针
    D: tl.constexpr,    # 每个 token 的 KV 维度 (num_heads * head_dim)
):
    idx = tl.program_id(0)                      # 当前 token 索引
    slot = tl.load(slot_mapping_ptr + idx)      # 目标槽位
    if slot == -1: return                       # 无效位置跳过
    
    # 加载 K/V
    key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))
    value = tl.load(value_ptr + idx * value_stride + tl.arange(0, D))
    
    # 写入 Cache
    tl.store(k_cache_ptr + slot * D + tl.arange(0, D), key)
    tl.store(v_cache_ptr + slot * D + tl.arange(0, D), value)
```

**使用 Triton 的原因**：
- slot_mapping 指定的位置不连续，PyTorch 索引操作效率低
- Triton 可以并行处理所有 token，减少 Kernel 启动开销

#### 3.6.2 `store_kvcache()` Python 封装

```python
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping, D
    )
```

#### 3.6.3 `Attention.forward()`

根据阶段选择不同的 flash_attn 函数。

```python
def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache
    
    # 1. 写入 KV Cache
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
    
    # 2. 计算 Attention
    if context.is_prefill:
        if context.block_tables is not None:  # Prefix Cache
            k, v = k_cache, v_cache
        o = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            block_table=context.block_tables
        )
    else:  # Decode
        o = flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache, v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=self.scale,
            causal=True
        )
    return o
```

**两种 flash_attn 函数对比**：

| 函数 | 适用阶段 | Q 形状 | KV 来源 | 特点 |
|------|----------|--------|---------|------|
| `flash_attn_varlen_func` | Prefill | 变长拼接 | 输入或 Cache | 支持变长序列批处理 |
| `flash_attn_with_kvcache` | Decode | [B, 1, H, D] | 仅 Cache | 针对单 token 优化 |

### 3.7 Scheduler 中的 Block 相关调用（scheduler.py）

Scheduler 是调度的核心，决定何时调用 BlockManager 的各个方法。

```python
class Scheduler:
    def __init__(self, config: Config):
        self.block_manager = BlockManager(config.num_kvcache_blocks, 
                                          config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()  # 等待队列
        self.running: deque[Sequence] = deque()  # 运行队列
```

**方法调用时机**：

| 方法 | 调用位置 | 触发条件 |
|------|----------|----------|
| `can_allocate` | `schedule()` | Prefill 阶段检查 |
| `allocate` | `schedule()` | 新请求开始执行 |
| `can_append` | `schedule()` | Decode 阶段检查 |
| `may_append` | `schedule()` | Decode 阶段每次迭代 |
| `deallocate` | `preempt()` | 抢占时释放 |
| `deallocate` | `postprocess()` | 请求完成时释放 |

---

## 四、Prefill 流程全解析

### 4.1 流程概述

Prefill 阶段处理新请求的 prompt，一次性计算所有 prompt token 的 KV Cache 并生成第一个 token。

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .prefill-container {
            background-color: white;
            padding: 30px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .prefill-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 25px;
            text-align: center;
        }
        .prefill-flow {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .prefill-step {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            width: 100%;
            max-width: 700px;
        }
        .prefill-num {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: #1976D2;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
            flex-shrink: 0;
        }
        .prefill-box {
            flex: 1;
            padding: 12px 18px;
            border: 2px solid #333;
            border-radius: 8px;
            font-size: 13px;
        }
        .prefill-module {
            font-weight: bold;
            color: #1976D2;
        }
        .prefill-method {
            font-family: 'Consolas', monospace;
            background: #E3F2FD;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }
        .prefill-arrow {
            font-size: 20px;
            color: #666;
            margin: 5px 0;
        }
        .prefill-detail {
            color: #666;
            font-size: 12px;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="prefill-container">
        <div class="prefill-title">Prefill 阶段函数调用链</div>
        <div class="prefill-flow">
            <div class="prefill-step">
                <div class="prefill-num">1</div>
                <div class="prefill-box" style="background: #E3F2FD;">
                    <span class="prefill-module">LLMEngine</span>.<span class="prefill-method">add_request(prompt, params)</span>
                    <div class="prefill-detail">创建 Sequence，加入 waiting 队列</div>
                </div>
            </div>
            <div class="prefill-arrow">↓</div>
            <div class="prefill-step">
                <div class="prefill-num">2</div>
                <div class="prefill-box" style="background: #FFF3E0;">
                    <span class="prefill-module">Scheduler</span>.<span class="prefill-method">schedule()</span>
                    <div class="prefill-detail">检查 can_allocate → 调用 allocate → 返回 (seqs, is_prefill=True)</div>
                </div>
            </div>
            <div class="prefill-arrow">↓</div>
            <div class="prefill-step">
                <div class="prefill-num">3</div>
                <div class="prefill-box" style="background: #FCE4EC;">
                    <span class="prefill-module">ModelRunner</span>.<span class="prefill-method">prepare_prefill(seqs)</span>
                    <div class="prefill-detail">构造 input_ids, positions, slot_mapping, cu_seqlens → set_context</div>
                </div>
            </div>
            <div class="prefill-arrow">↓</div>
            <div class="prefill-step">
                <div class="prefill-num">4</div>
                <div class="prefill-box" style="background: #E0F7FA;">
                    <span class="prefill-module">Attention</span>.<span class="prefill-method">forward(q, k, v)</span>
                    <div class="prefill-detail">store_kvcache 写入 → flash_attn_varlen_func 计算</div>
                </div>
            </div>
            <div class="prefill-arrow">↓</div>
            <div class="prefill-step">
                <div class="prefill-num">5</div>
                <div class="prefill-box" style="background: #E8F5E9;">
                    <span class="prefill-module">Sampler</span>.<span class="prefill-method">forward(logits, temperatures)</span>
                    <div class="prefill-detail">采样生成第一个 token</div>
                </div>
            </div>
            <div class="prefill-arrow">↓</div>
            <div class="prefill-step">
                <div class="prefill-num">6</div>
                <div class="prefill-box" style="background: #F3E5F5;">
                    <span class="prefill-module">Scheduler</span>.<span class="prefill-method">postprocess(seqs, token_ids)</span>
                    <div class="prefill-detail">追加 token 到 Sequence，检查终止条件</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
```

### 4.2 Step 1：请求进入调度（Scheduler.schedule）

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0
    
    # Prefill 调度
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        
        # 检查 token 数量限制
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
            break
        
        # 检查 Block 是否足够
        if not self.block_manager.can_allocate(seq):
            break
        
        # 分配 Block
        num_seqs += 1
        self.block_manager.allocate(seq)
        
        # 更新计数（只计算非缓存 token）
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        
        # 状态转移
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    
    if scheduled_seqs:
        return scheduled_seqs, True  # is_prefill = True
    
    # 无 Prefill 请求，尝试 Decode...
```

**Prefix Caching 的命中处理**：

在 `allocate` 内部，如果检测到缓存命中：
1. 复用已有 Block，增加其 `ref_count`
2. 累加 `seq.num_cached_tokens`
3. 后续 `prepare_prefill` 会利用这个信息跳过缓存 token

### 4.3 Step 2：准备运行时上下文（ModelRunner.prepare_prefill）

```python
def prepare_prefill(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    block_tables = None
    
    for seq in seqs:
        seqlen = len(seq)
        
        # 只包含非缓存的 token
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(list(range(seq.num_cached_tokens, seqlen)))
        
        # 计算序列长度
        seqlen_q = seqlen - seq.num_cached_tokens  # Query 长度（非缓存）
        seqlen_k = seqlen                           # Key 长度（全部）
        
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)
        
        # 构造 slot_mapping（只为非缓存 token）
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            start = seq.block_table[i] * self.block_size
            end = start + (self.block_size if i != seq.num_blocks - 1 
                          else seq.last_block_num_tokens)
            slot_mapping.extend(list(range(start, end)))
    
    # 如果有 Prefix Cache 命中，需要传递 block_tables
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
        block_tables = self.prepare_block_tables(seqs)
    
    # 转换为 Tensor
    input_ids = torch.tensor(input_ids, dtype=torch.int64).cuda()
    positions = torch.tensor(positions, dtype=torch.int64).cuda()
    # ... 其他转换 ...
    
    # 设置上下文
    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                slot_mapping, None, block_tables)
    
    return input_ids, positions
```

**slot_mapping 构造图示**：

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .slot-container {
            background-color: white;
            padding: 30px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .slot-title {
            font-size: 15px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .slot-row {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
        }
        .slot-label {
            width: 130px;
            font-size: 12px;
            color: #555;
            font-weight: bold;
        }
        .slot-cells {
            display: flex;
        }
        .slot-cell {
            width: 38px;
            height: 32px;
            border: 1px solid #666;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            margin-right: 2px;
            border-radius: 3px;
        }
        .slot-cached { background: #E0E0E0; color: #999; }
        .slot-new { background: #C8E6C9; }
        .slot-result {
            margin-top: 20px;
            padding: 12px 15px;
            background: #FFF3E0;
            border-radius: 5px;
            font-family: 'Consolas', monospace;
            font-size: 12px;
        }
        .slot-note {
            font-size: 11px;
            color: #888;
            margin-top: 12px;
        }
    </style>
</head>
<body>
    <div class="slot-container">
        <div class="slot-title">Prefill slot_mapping 构造示例（block_size=4, 2个Block缓存命中）</div>
        
        <div class="slot-row">
            <div class="slot-label">Token 序列：</div>
            <div class="slot-cells">
                <div class="slot-cell slot-cached">t0</div>
                <div class="slot-cell slot-cached">t1</div>
                <div class="slot-cell slot-cached">t2</div>
                <div class="slot-cell slot-cached">t3</div>
                <div class="slot-cell slot-cached">t4</div>
                <div class="slot-cell slot-cached">t5</div>
                <div class="slot-cell slot-cached">t6</div>
                <div class="slot-cell slot-cached">t7</div>
                <div class="slot-cell slot-new">t8</div>
                <div class="slot-cell slot-new">t9</div>
                <div class="slot-cell slot-new">t10</div>
                <div class="slot-cell slot-new">t11</div>
                <div class="slot-cell slot-new">t12</div>
                <div class="slot-cell slot-new">t13</div>
            </div>
        </div>
        
        <div class="slot-row">
            <div class="slot-label">Block 划分：</div>
            <div class="slot-cells">
                <div style="width: 160px; text-align: center; font-size: 11px; color: #999;">Block 0-1 (缓存命中)</div>
                <div style="width: 80px; text-align: center; font-size: 11px; color: #4CAF50;">Block 2 (新)</div>
                <div style="width: 80px; text-align: center; font-size: 11px; color: #4CAF50;">Block 3 (新)</div>
            </div>
        </div>
        
        <div class="slot-row">
            <div class="slot-label">block_table：</div>
            <div style="font-size: 12px;">[5, 8, 3, 12] (物理 Block ID)</div>
        </div>
        
        <div class="slot-result">
            <strong>slot_mapping 计算过程：</strong><br><br>
            Block 2 (physical=3): slots = [3×4, 3×4+1, 3×4+2, 3×4+3] = [12, 13, 14, 15]<br>
            Block 3 (physical=12): slots = [12×4, 12×4+1] = [48, 49]<br><br>
            <strong>最终：</strong> slot_mapping = [12, 13, 14, 15, 48, 49]
        </div>
        
        <div class="slot-note">
            * 灰色 token (t0-t7) 已缓存，不在 input_ids 中，不生成 slot_mapping<br>
            * input_ids = [t8, t9, t10, t11, t12, t13]，长度为 6
        </div>
    </div>
</body>
</html>
```

### 4.4 Step 3：执行 Attention（Attention.forward）

```python
def forward(self, q, k, v):
    context = get_context()
    
    # 1. 写入 KV Cache
    store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
    
    # 2. 计算 Attention
    if context.block_tables is not None:
        # Prefix Cache 命中：从 Cache 读取历史 KV
        k, v = self.k_cache, self.v_cache
    
    o = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=context.cu_seqlens_q,
        cu_seqlens_k=context.cu_seqlens_k,
        max_seqlen_q=context.max_seqlen_q,
        max_seqlen_k=context.max_seqlen_k,
        softmax_scale=self.scale,
        causal=True,
        block_table=context.block_tables
    )
    return o
```

**Prefix Cache 命中时的特殊处理**：

当发生 Prefix Cache 命中时：
- `cu_seqlens_q`（Query 累积长度）< `cu_seqlens_k`（Key 累积长度）
- 传入的 `k, v` 只包含新计算的 token
- 通过 `block_tables` 参数，flash_attn 会从 Cache 中读取历史 KV
- 内部拼接历史 KV 和新 KV 进行 Attention 计算

### 4.5 Step 4：采样与状态更新

```python
# ModelRunner.run
logits = self.run_model(input_ids, positions, is_prefill)
token_ids = self.sampler(logits, temperatures).tolist()

# Scheduler.postprocess
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)  # 追加新 token
        
        # 检查终止条件
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

### 4.6 完整流程图

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .flow-container {
            background-color: white;
            padding: 30px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .flow-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 25px;
            text-align: center;
        }
        .flow-diagram {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        .flow-column {
            width: 30%;
        }
        .flow-col-title {
            font-weight: bold;
            text-align: center;
            padding: 8px;
            margin-bottom: 15px;
            border-radius: 5px;
        }
        .flow-item {
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 11px;
            text-align: center;
        }
        .flow-arrow-h {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 5px;
            color: #666;
        }
        .flow-data {
            font-size: 10px;
            color: #666;
            text-align: center;
            padding: 5px;
            background: #FAFAFA;
            margin: 5px 0;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="flow-container">
        <div class="flow-title">Prefill 阶段数据流转图</div>
        <div class="flow-diagram">
            <div class="flow-column">
                <div class="flow-col-title" style="background: #E3F2FD;">调度层</div>
                <div class="flow-item" style="background: #E3F2FD;">
                    <strong>Scheduler.schedule()</strong><br>
                    从 waiting 取出 Sequence
                </div>
                <div class="flow-arrow-h">↓</div>
                <div class="flow-item" style="background: #FFF3E0;">
                    <strong>BlockManager.allocate()</strong><br>
                    分配 Block，更新 block_table
                </div>
                <div class="flow-data">
                    block_table = [7, 3, 12]<br>
                    num_cached_tokens = 512
                </div>
            </div>
            
            <div class="flow-arrow-h" style="font-size: 24px; padding-top: 80px;">→</div>
            
            <div class="flow-column">
                <div class="flow-col-title" style="background: #FCE4EC;">执行层</div>
                <div class="flow-item" style="background: #FCE4EC;">
                    <strong>ModelRunner.prepare_prefill()</strong><br>
                    构造 slot_mapping, cu_seqlens
                </div>
                <div class="flow-data">
                    input_ids: [N_new] tokens<br>
                    slot_mapping: [N_new] slots
                </div>
                <div class="flow-arrow-h">↓</div>
                <div class="flow-item" style="background: #E0F7FA;">
                    <strong>Attention.forward()</strong><br>
                    store_kvcache + flash_attn
                </div>
                <div class="flow-data">
                    KV Cache 写入完成<br>
                    输出 hidden_states
                </div>
            </div>
            
            <div class="flow-arrow-h" style="font-size: 24px; padding-top: 80px;">→</div>
            
            <div class="flow-column">
                <div class="flow-col-title" style="background: #E8F5E9;">输出层</div>
                <div class="flow-item" style="background: #E8F5E9;">
                    <strong>LMHead + Sampler</strong><br>
                    计算 logits，采样 token
                </div>
                <div class="flow-data">
                    new_token_id = 12345
                </div>
                <div class="flow-arrow-h">↓</div>
                <div class="flow-item" style="background: #F3E5F5;">
                    <strong>Scheduler.postprocess()</strong><br>
                    追加 token，检查终止
                </div>
                <div class="flow-data">
                    seq.num_tokens += 1<br>
                    进入 Decode 阶段
                </div>
            </div>
        </div>
    </div>
</body>
</html>
```

---

## 五、Decode 流程全解析

### 5.1 流程概述

Decode 阶段逐个生成 token，每次迭代只处理一个 token，但可以批量处理多个 Sequence。

**与 Prefill 的关键差异**：

| 方面 | Prefill | Decode |
|------|---------|--------|
| Token 数量 | 多个（整个 prompt） | 每个 Sequence 1 个 |
| Block 操作 | allocate（分配多个） | may_append（可能追加 1 个） |
| Attention 函数 | flash_attn_varlen_func | flash_attn_with_kvcache |
| slot_mapping | 多个 slot | 每个 Sequence 1 个 slot |
| 资源不足处理 | 跳过等待 | 抢占其他请求 |

### 5.2 Step 1：调度与 Block 追加（Scheduler.schedule）

```python
def schedule(self):
    # ... Prefill 调度返回空时，进入 Decode 调度 ...
    
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()
        
        # 检查是否能追加
        while not self.block_manager.can_append(seq):
            # 资源不足，需要抢占
            if self.running:
                self.preempt(self.running.pop())  # 抢占最后进入的
            else:
                self.preempt(seq)  # 无法继续，抢占自己
                break
        else:
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False  # is_prefill = False
```

**`may_append` 的三种情况**：

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .append-container {
            background-color: white;
            padding: 30px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .append-title {
            font-size: 15px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .append-case {
            margin-bottom: 25px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .append-case-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .append-blocks {
            display: flex;
            align-items: center;
            margin: 10px 0;
        }
        .append-block {
            width: 80px;
            height: 50px;
            border: 2px solid #333;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-right: 10px;
            font-size: 11px;
            border-radius: 4px;
        }
        .append-full { background: #C8E6C9; }
        .append-partial { background: #FFF9C4; }
        .append-new { background: #BBDEFB; border-style: dashed; }
        .append-slot {
            width: 15px;
            height: 15px;
            border: 1px solid #666;
            margin: 1px;
            display: inline-block;
            font-size: 8px;
            text-align: center;
            line-height: 15px;
        }
        .append-slot-used { background: #81C784; }
        .append-slot-new { background: #FFD54F; }
        .append-slot-empty { background: #eee; }
        .append-action {
            font-size: 12px;
            color: #666;
            margin-top: 10px;
            padding: 8px;
            background: #F5F5F5;
            border-radius: 4px;
        }
        .append-code {
            font-family: 'Consolas', monospace;
            background: #ECEFF1;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="append-container">
        <div class="append-title">may_append 的三种情况（假设 block_size = 4）</div>
        
        <div class="append-case">
            <div class="append-case-title">情况 1：需要新 Block（len % block_size == 1）</div>
            <div class="append-blocks">
                <div class="append-block append-full">
                    Block 0<br>
                    <div>
                        <span class="append-slot append-slot-used">✓</span>
                        <span class="append-slot append-slot-used">✓</span>
                        <span class="append-slot append-slot-used">✓</span>
                        <span class="append-slot append-slot-used">✓</span>
                    </div>
                </div>
                <div class="append-block append-new">
                    Block 1 (新)<br>
                    <div>
                        <span class="append-slot append-slot-new">★</span>
                        <span class="append-slot append-slot-empty"></span>
                        <span class="append-slot append-slot-empty"></span>
                        <span class="append-slot append-slot-empty"></span>
                    </div>
                </div>
            </div>
            <div class="append-action">
                <span class="append-code">len(seq) = 5</span>，上一个 Block 已满<br>
                操作：从 free_block_ids 分配新 Block，追加到 block_table
            </div>
        </div>
        
        <div class="append-case">
            <div class="append-case-title">情况 2：Block 刚填满（len % block_size == 0）</div>
            <div class="append-blocks">
                <div class="append-block append-full">
                    Block 0<br>
                    <div>
                        <span class="append-slot append-slot-used">✓</span>
                        <span class="append-slot append-slot-used">✓</span>
                        <span class="append-slot append-slot-used">✓</span>
                        <span class="append-slot append-slot-new">★</span>
                    </div>
                </div>
            </div>
            <div class="append-action">
                <span class="append-code">len(seq) = 4</span>，Block 0 刚好填满<br>
                操作：计算 Block 0 的 hash，注册到 hash_to_block_id（供后续 Prefix Cache）
            </div>
        </div>
        
        <div class="append-case">
            <div class="append-case-title">情况 3：Block 正在填充（其他情况）</div>
            <div class="append-blocks">
                <div class="append-block append-partial">
                    Block 0<br>
                    <div>
                        <span class="append-slot append-slot-used">✓</span>
                        <span class="append-slot append-slot-used">✓</span>
                        <span class="append-slot append-slot-new">★</span>
                        <span class="append-slot append-slot-empty"></span>
                    </div>
                </div>
            </div>
            <div class="append-action">
                <span class="append-code">len(seq) = 3</span>，Block 0 还有空位<br>
                操作：无需任何操作，新 token 直接写入现有 Block
            </div>
        </div>
    </div>
</body>
</html>
```

### 5.3 Step 2：准备运行时上下文（ModelRunner.prepare_decode）

```python
def prepare_decode(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []
    
    for seq in seqs:
        # 只取最后一个 token
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        
        # 计算新 token 的 slot
        slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        slot_mapping.append(slot)
    
    # 转换为 Tensor
    input_ids = torch.tensor(input_ids, dtype=torch.int64).cuda()
    positions = torch.tensor(positions, dtype=torch.int64).cuda()
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).cuda()
    context_lens = torch.tensor(context_lens, dtype=torch.int32).cuda()
    
    # 构造批量 block_tables
    block_tables = self.prepare_block_tables(seqs)
    
    # 设置上下文
    set_context(False, slot_mapping=slot_mapping, 
                context_lens=context_lens, block_tables=block_tables)
    
    return input_ids, positions
```

**Decode slot_mapping 示意**：

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .decode-slot-container {
            background-color: white;
            padding: 25px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .decode-slot-title {
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .decode-slot-row {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .decode-slot-label {
            width: 120px;
            font-size: 12px;
            font-weight: bold;
        }
        .decode-slot-value {
            font-size: 12px;
        }
        .decode-slot-formula {
            margin-top: 15px;
            padding: 12px;
            background: #E3F2FD;
            border-radius: 5px;
            font-family: 'Consolas', monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="decode-slot-container">
        <div class="decode-slot-title">Decode 阶段 slot_mapping 计算（单 token）</div>
        
        <div class="decode-slot-row">
            <div class="decode-slot-label">假设条件：</div>
            <div class="decode-slot-value">len(seq) = 11, block_size = 4, block_table = [7, 3, 12]</div>
        </div>
        
        <div class="decode-slot-row">
            <div class="decode-slot-label">最后一个 Block：</div>
            <div class="decode-slot-value">block_table[-1] = 12</div>
        </div>
        
        <div class="decode-slot-row">
            <div class="decode-slot-label">Block 内位置：</div>
            <div class="decode-slot-value">last_block_num_tokens = 11 - 2×4 = 3（第 3 个槽位，索引为 2）</div>
        </div>
        
        <div class="decode-slot-formula">
            slot = block_table[-1] × block_size + last_block_num_tokens - 1<br>
            &nbsp;&nbsp;&nbsp;&nbsp; = 12 × 4 + 3 - 1<br>
            &nbsp;&nbsp;&nbsp;&nbsp; = <strong>50</strong>
        </div>
    </div>
</body>
</html>
```

### 5.4 Step 3：执行 Attention（Attention.forward）

```python
def forward(self, q, k, v):
    context = get_context()
    
    # 1. 写入单个 token 的 KV
    store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
    
    # 2. Decode 阶段使用 flash_attn_with_kvcache
    o = flash_attn_with_kvcache(
        q.unsqueeze(1),  # [batch, 1, heads, dim]
        self.k_cache,
        self.v_cache,
        cache_seqlens=context.context_lens,  # 每个序列的长度
        block_table=context.block_tables,     # 批量 block_table
        softmax_scale=self.scale,
        causal=True
    )
    return o
```

**flash_attn_with_kvcache 的工作方式**：
1. 根据 `block_tables` 定位每个 Sequence 的历史 KV 在 Cache 中的位置
2. 根据 `cache_seqlens` 确定每个 Sequence 的有效长度
3. 对每个新 Query token，与其对应的所有历史 KV 计算 Attention

### 5.5 Step 4：后处理（Scheduler.postprocess）

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
    for seq, token_id in zip(seqs, token_ids):
        # 追加新 token
        seq.append_token(token_id)
        
        # 检查终止条件
        is_eos = not seq.ignore_eos and token_id == self.eos
        is_max_tokens = seq.num_completion_tokens == seq.max_tokens
        
        if is_eos or is_max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)  # 释放所有 Block
            self.running.remove(seq)
```

### 5.6 完整流程图与对比

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .compare-container {
            background-color: white;
            padding: 30px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .compare-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 25px;
            text-align: center;
        }
        .compare-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .compare-table th, .compare-table td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        .compare-table th {
            background: #F5F5F5;
            font-weight: bold;
        }
        .compare-prefill { background: #E3F2FD; }
        .compare-decode { background: #E8F5E9; }
        .compare-code {
            font-family: 'Consolas', monospace;
            background: #ECEFF1;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="compare-container">
        <div class="compare-title">Prefill vs Decode 流程对比</div>
        <table class="compare-table">
            <tr>
                <th style="width: 20%;">步骤</th>
                <th style="width: 40%;" class="compare-prefill">Prefill</th>
                <th style="width: 40%;" class="compare-decode">Decode</th>
            </tr>
            <tr>
                <td><strong>Block 操作</strong></td>
                <td class="compare-prefill">
                    <span class="compare-code">allocate(seq)</span><br>
                    一次性分配所有需要的 Block
                </td>
                <td class="compare-decode">
                    <span class="compare-code">may_append(seq)</span><br>
                    按需追加（可能 0 或 1 个 Block）
                </td>
            </tr>
            <tr>
                <td><strong>input_ids</strong></td>
                <td class="compare-prefill">
                    整个 prompt（减去缓存部分）<br>
                    形状: [total_tokens]
                </td>
                <td class="compare-decode">
                    每个 Sequence 1 个 token<br>
                    形状: [batch_size]
                </td>
            </tr>
            <tr>
                <td><strong>slot_mapping</strong></td>
                <td class="compare-prefill">
                    所有非缓存 token 的 slot<br>
                    长度 = total_tokens
                </td>
                <td class="compare-decode">
                    每个 Sequence 1 个 slot<br>
                    长度 = batch_size
                </td>
            </tr>
            <tr>
                <td><strong>Attention 函数</strong></td>
                <td class="compare-prefill">
                    <span class="compare-code">flash_attn_varlen_func</span><br>
                    处理变长序列拼接
                </td>
                <td class="compare-decode">
                    <span class="compare-code">flash_attn_with_kvcache</span><br>
                    针对单 token + Cache 优化
                </td>
            </tr>
            <tr>
                <td><strong>Context 字段</strong></td>
                <td class="compare-prefill">
                    cu_seqlens_q/k, max_seqlen_q/k<br>
                    block_tables（仅 Prefix Cache）
                </td>
                <td class="compare-decode">
                    context_lens, block_tables
                </td>
            </tr>
            <tr>
                <td><strong>资源不足处理</strong></td>
                <td class="compare-prefill">
                    跳过该请求，等待资源
                </td>
                <td class="compare-decode">
                    抢占其他请求释放资源
                </td>
            </tr>
        </table>
    </div>
</body>
</html>
```

---

## 六、Prefix Caching 机制详解

### 6.1 设计动机

在实际应用中，大量请求共享相同的前缀（如 System Prompt）：

```
请求1: [System Prompt] + "What is AI?"
请求2: [System Prompt] + "Explain ML"
请求3: [System Prompt] + "Write code"
```

如果每个请求都重新计算 System Prompt 的 KV Cache，会造成大量的计算和显存浪费。**Prefix Caching** 允许这些请求共享相同前缀的 Block。

### 6.2 Hash 计算与匹配

nano-vllm 使用 **链式 hash** 来标识 Block 内容：

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))  # 包含前缀 hash
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

**链式 hash 的原理**：

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .hash-container {
            background-color: white;
            padding: 30px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .hash-title {
            font-size: 15px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .hash-chain {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
        .hash-block {
            width: 120px;
            padding: 12px;
            border: 2px solid #333;
            border-radius: 8px;
            text-align: center;
            margin-right: 15px;
        }
        .hash-block-title {
            font-weight: bold;
            font-size: 13px;
            margin-bottom: 8px;
        }
        .hash-content {
            font-size: 11px;
            color: #666;
            margin-bottom: 8px;
        }
        .hash-value {
            font-family: 'Consolas', monospace;
            font-size: 10px;
            background: #E3F2FD;
            padding: 3px 6px;
            border-radius: 3px;
        }
        .hash-arrow {
            font-size: 20px;
            color: #666;
            margin-right: 15px;
        }
        .hash-formula {
            margin-top: 20px;
            padding: 15px;
            background: #FFF3E0;
            border-radius: 5px;
            font-size: 12px;
        }
        .hash-note {
            margin-top: 15px;
            font-size: 11px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="hash-container">
        <div class="hash-title">链式 Hash 计算示意</div>
        
        <div class="hash-chain">
            <div class="hash-block" style="background: #C8E6C9;">
                <div class="hash-block-title">Block 0</div>
                <div class="hash-content">tokens: [t0-t255]</div>
                <div class="hash-value">h0 = hash(tokens)</div>
            </div>
            <div class="hash-arrow">→</div>
            <div class="hash-block" style="background: #C8E6C9;">
                <div class="hash-block-title">Block 1</div>
                <div class="hash-content">tokens: [t256-t511]</div>
                <div class="hash-value">h1 = hash(h0, tokens)</div>
            </div>
            <div class="hash-arrow">→</div>
            <div class="hash-block" style="background: #C8E6C9;">
                <div class="hash-block-title">Block 2</div>
                <div class="hash-content">tokens: [t512-t767]</div>
                <div class="hash-value">h2 = hash(h1, tokens)</div>
            </div>
            <div class="hash-arrow">→</div>
            <div class="hash-block" style="background: #FFF9C4; border-style: dashed;">
                <div class="hash-block-title">Block 3</div>
                <div class="hash-content">tokens: [t768-t800]</div>
                <div class="hash-value">h3 = -1 (未满)</div>
            </div>
        </div>
        
        <div class="hash-formula">
            <strong>计算规则：</strong><br><br>
            • Block 0: h₀ = xxhash(tokens₀)<br>
            • Block n: hₙ = xxhash(hₙ₋₁ || tokensₙ)<br>
            • 未填满的 Block: hash = -1（不参与缓存）
        </div>
        
        <div class="hash-note">
            <strong>为什么需要链式 hash？</strong><br>
            确保只有<strong>完全相同的前缀</strong>才能匹配。即使 Block 2 的 tokens 相同，
            如果 Block 0 或 Block 1 不同，h2 也会不同。
        </div>
    </div>
</body>
</html>
```

**双重校验的必要性**：

```python
# 仅 hash 匹配不够，还需校验内容
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    cache_miss = True
```

原因：
1. **Hash 碰撞**：xxhash 碰撞概率极低但不为零
2. **Block 被覆写**：Block 回收后重新分配，旧的 hash→block_id 映射可能未清除

### 6.3 在 Prefill 流程中的体现

当新请求的前缀与已有 Block 匹配时：

```python
def allocate(self, seq: Sequence):
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)
        
        if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
            # Cache Hit!
            seq.num_cached_tokens += self.block_size
            self.blocks[block_id].ref_count += 1
            # 不需要从 free_block_ids 分配
        else:
            # Cache Miss
            cache_miss = True
            # 从 free_block_ids 分配新 Block
```

**后续影响**：

1. `seq.num_cached_tokens` 累加了缓存命中的 token 数
2. `prepare_prefill` 中 `input_ids` 只包含 `seq[seq.num_cached_tokens:]`
3. `slot_mapping` 只为非缓存 token 生成
4. Attention 计算时通过 `block_tables` 从 Cache 读取历史 KV

### 6.4 Block 共享与引用计数

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .ref-container {
            background-color: white;
            padding: 30px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .ref-title {
            font-size: 15px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .ref-timeline {
            position: relative;
            padding-left: 30px;
        }
        .ref-event {
            position: relative;
            padding: 15px;
            margin-bottom: 15px;
            background: #FAFAFA;
            border-left: 3px solid #2196F3;
            border-radius: 0 5px 5px 0;
        }
        .ref-event::before {
            content: '';
            position: absolute;
            left: -9px;
            top: 18px;
            width: 12px;
            height: 12px;
            background: #2196F3;
            border-radius: 50%;
        }
        .ref-event-title {
            font-weight: bold;
            margin-bottom: 8px;
        }
        .ref-event-detail {
            font-size: 12px;
            color: #666;
        }
        .ref-block-state {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="ref-container">
        <div class="ref-title">ref_count 变化时间线</div>
        
        <div class="ref-timeline">
            <div class="ref-event">
                <div class="ref-event-title">T1: Sequence A 到达（包含 System Prompt）</div>
                <div class="ref-event-detail">
                    Block 5 被分配，存储 System Prompt
                    <span class="ref-block-state" style="background: #C8E6C9;">Block 5: ref_count = 1</span>
                </div>
            </div>
            
            <div class="ref-event">
                <div class="ref-event-title">T2: Sequence B 到达（相同 System Prompt）</div>
                <div class="ref-event-detail">
                    检测到 Block 5 的 hash 匹配，复用该 Block
                    <span class="ref-block-state" style="background: #FFF9C4;">Block 5: ref_count = 2</span>
                </div>
            </div>
            
            <div class="ref-event">
                <div class="ref-event-title">T3: Sequence C 到达（相同 System Prompt）</div>
                <div class="ref-event-detail">
                    继续复用 Block 5
                    <span class="ref-block-state" style="background: #FFCCBC;">Block 5: ref_count = 3</span>
                </div>
            </div>
            
            <div class="ref-event">
                <div class="ref-event-title">T4: Sequence A 完成</div>
                <div class="ref-event-detail">
                    调用 deallocate(A)，Block 5 的 ref_count 减 1
                    <span class="ref-block-state" style="background: #FFF9C4;">Block 5: ref_count = 2</span>
                    <br><small>Block 5 不会被释放，因为 ref_count > 0</small>
                </div>
            </div>
            
            <div class="ref-event">
                <div class="ref-event-title">T5: Sequence B, C 都完成</div>
                <div class="ref-event-detail">
                    ref_count 降为 0，Block 5 归还到 free_block_ids
                    <span class="ref-block-state" style="background: #E0E0E0;">Block 5: ref_count = 0 (freed)</span>
                    <br><small>但 hash→block_id 映射保留，下次可能命中</small>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
```

### 6.5 示例演示

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .pc-demo-container {
            background-color: white;
            padding: 30px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .pc-demo-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 25px;
        }
        .pc-demo-seq {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .pc-demo-seq-title {
            font-weight: bold;
            margin-bottom: 15px;
            font-size: 14px;
        }
        .pc-demo-blocks {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .pc-demo-block {
            width: 100px;
            padding: 10px;
            border: 2px solid #333;
            border-radius: 6px;
            text-align: center;
            font-size: 11px;
        }
        .pc-demo-shared { background: #C8E6C9; border-color: #4CAF50; }
        .pc-demo-unique { background: #BBDEFB; }
        .pc-demo-arrow {
            font-size: 16px;
            color: #4CAF50;
        }
        .pc-demo-status {
            font-size: 10px;
            margin-top: 5px;
            padding: 2px 5px;
            border-radius: 3px;
        }
        .pc-demo-physical {
            margin-top: 25px;
            padding: 15px;
            background: #FAFAFA;
            border-radius: 5px;
        }
        .pc-demo-physical-title {
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 13px;
        }
        .pc-demo-physical-blocks {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }
        .pc-demo-physical-block {
            width: 50px;
            height: 40px;
            border: 1px solid #999;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            border-radius: 3px;
        }
        .pc-demo-summary {
            margin-top: 20px;
            padding: 15px;
            background: #E3F2FD;
            border-radius: 5px;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="pc-demo-container">
        <div class="pc-demo-title">Prefix Caching 完整示例</div>
        
        <div class="pc-demo-seq">
            <div class="pc-demo-seq-title">Sequence 1: "You are a helpful assistant." + "What is AI?"</div>
            <div class="pc-demo-blocks">
                <div class="pc-demo-block pc-demo-shared">
                    Block 0<br>
                    "You are a..."<br>
                    <span class="pc-demo-status" style="background: #A5D6A7;">Physical: 5</span>
                </div>
                <div class="pc-demo-block pc-demo-shared">
                    Block 1<br>
                    "...helpful..."<br>
                    <span class="pc-demo-status" style="background: #A5D6A7;">Physical: 8</span>
                </div>
                <div class="pc-demo-block pc-demo-unique">
                    Block 2<br>
                    "What is AI?"<br>
                    <span class="pc-demo-status" style="background: #90CAF9;">Physical: 3</span>
                </div>
            </div>
            <div style="font-size: 11px; color: #666; margin-top: 10px;">
                block_table = [5, 8, 3], num_cached_tokens = 0（首次请求，无缓存）
            </div>
        </div>
        
        <div class="pc-demo-seq">
            <div class="pc-demo-seq-title">Sequence 2: "You are a helpful assistant." + "Explain ML"（稍后到达）</div>
            <div class="pc-demo-blocks">
                <div class="pc-demo-block pc-demo-shared">
                    Block 0<br>
                    "You are a..."<br>
                    <span class="pc-demo-status" style="background: #A5D6A7;">Physical: 5 ✓</span>
                </div>
                <span class="pc-demo-arrow">← 缓存命中!</span>
                <div class="pc-demo-block pc-demo-shared">
                    Block 1<br>
                    "...helpful..."<br>
                    <span class="pc-demo-status" style="background: #A5D6A7;">Physical: 8 ✓</span>
                </div>
                <span class="pc-demo-arrow">← 缓存命中!</span>
                <div class="pc-demo-block pc-demo-unique">
                    Block 2<br>
                    "Explain ML"<br>
                    <span class="pc-demo-status" style="background: #90CAF9;">Physical: 12</span>
                </div>
            </div>
            <div style="font-size: 11px; color: #666; margin-top: 10px;">
                block_table = [5, 8, 12], num_cached_tokens = 512（2个Block缓存命中）
            </div>
        </div>
        
        <div class="pc-demo-physical">
            <div class="pc-demo-physical-title">物理 Block 状态：</div>
            <div class="pc-demo-physical-blocks">
                <div class="pc-demo-physical-block" style="background: #eee;">0<br>ref=0</div>
                <div class="pc-demo-physical-block" style="background: #eee;">1<br>ref=0</div>
                <div class="pc-demo-physical-block" style="background: #eee;">2<br>ref=0</div>
                <div class="pc-demo-physical-block" style="background: #BBDEFB;">3<br>ref=1</div>
                <div class="pc-demo-physical-block" style="background: #eee;">4<br>ref=0</div>
                <div class="pc-demo-physical-block" style="background: #C8E6C9; border: 2px solid #4CAF50;">5<br>ref=2</div>
                <div class="pc-demo-physical-block" style="background: #eee;">6<br>ref=0</div>
                <div class="pc-demo-physical-block" style="background: #eee;">7<br>ref=0</div>
                <div class="pc-demo-physical-block" style="background: #C8E6C9; border: 2px solid #4CAF50;">8<br>ref=2</div>
                <div class="pc-demo-physical-block" style="background: #eee;">9<br>ref=0</div>
                <div class="pc-demo-physical-block" style="background: #eee;">10<br>ref=0</div>
                <div class="pc-demo-physical-block" style="background: #eee;">11<br>ref=0</div>
                <div class="pc-demo-physical-block" style="background: #BBDEFB;">12<br>ref=1</div>
            </div>
        </div>
        
        <div class="pc-demo-summary">
            <strong>收益分析：</strong><br><br>
            • Sequence 2 的 Prefill：只需计算 "Explain ML" 部分（约 256 tokens）<br>
            • 节省计算：512 tokens 的 KV 计算（2 个 Block）<br>
            • 节省显存：2 个 Block × 29.36 MB ≈ 59 MB<br>
            • 如果有 100 个类似请求，总节省约 5.9 GB 显存
        </div>
    </div>
</body>
</html>
```

---

## 七、抢占与恢复机制

### 7.1 抢占触发条件

当 Decode 阶段需要新 Block 但空闲池已空时，触发抢占：

```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())  # 抢占
    else:
        self.preempt(seq)
        break
```

### 7.2 抢占策略

nano-vllm 采用 **LIFO（后进先出）** 策略：抢占最后进入 running 队列的请求。

**选择 LIFO 的原因**：
1. 后进入的请求生成的 token 更少，重新计算代价更小
2. 先进入的请求接近完成，应优先保护
3. 实现简单：`running.pop()` 即可

### 7.3 抢占处理流程

```python
def preempt(self, seq: Sequence):
    # 1. 状态回退
    seq.status = SequenceStatus.WAITING
    
    # 2. 释放所有 Block
    self.block_manager.deallocate(seq)
    
    # 3. 放回等待队列头部（优先恢复）
    self.waiting.appendleft(seq)
```

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .preempt-container {
            background-color: white;
            padding: 30px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .preempt-title {
            font-size: 15px;
            font-weight: bold;
            margin-bottom: 25px;
        }
        .preempt-step {
            display: flex;
            margin-bottom: 20px;
        }
        .preempt-num {
            width: 30px;
            height: 30px;
            background: #F44336;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
            flex-shrink: 0;
        }
        .preempt-content {
            flex: 1;
        }
        .preempt-action {
            font-weight: bold;
            margin-bottom: 8px;
        }
        .preempt-detail {
            font-size: 12px;
            color: #666;
            padding: 10px;
            background: #FFEBEE;
            border-radius: 5px;
        }
        .preempt-queue {
            display: flex;
            align-items: center;
            margin-top: 25px;
            padding: 15px;
            background: #FAFAFA;
            border-radius: 5px;
        }
        .preempt-queue-title {
            width: 80px;
            font-weight: bold;
            font-size: 12px;
        }
        .preempt-queue-item {
            padding: 8px 15px;
            margin-right: 8px;
            border-radius: 5px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="preempt-container">
        <div class="preempt-title">抢占处理流程</div>
        
        <div class="preempt-step">
            <div class="preempt-num">1</div>
            <div class="preempt-content">
                <div class="preempt-action">检测资源不足</div>
                <div class="preempt-detail">
                    can_append(seq) 返回 False<br>
                    free_block_ids 为空，无法分配新 Block
                </div>
            </div>
        </div>
        
        <div class="preempt-step">
            <div class="preempt-num">2</div>
            <div class="preempt-content">
                <div class="preempt-action">选择抢占目标</div>
                <div class="preempt-detail">
                    running.pop() — 选择最后进入的 Sequence<br>
                    例如：Seq C 是最后加入的，被选中抢占
                </div>
            </div>
        </div>
        
        <div class="preempt-step">
            <div class="preempt-num">3</div>
            <div class="preempt-content">
                <div class="preempt-action">释放资源</div>
                <div class="preempt-detail">
                    block_manager.deallocate(seq_c)<br>
                    Seq C 的所有 Block 归还到 free_block_ids
                </div>
            </div>
        </div>
        
        <div class="preempt-step">
            <div class="preempt-num">4</div>
            <div class="preempt-content">
                <div class="preempt-action">状态回退</div>
                <div class="preempt-detail">
                    seq_c.status = WAITING<br>
                    waiting.appendleft(seq_c) — 放到等待队列头部
                </div>
            </div>
        </div>
        
        <div class="preempt-queue">
            <div class="preempt-queue-title">抢占前：</div>
            <div class="preempt-queue-item" style="background: #C8E6C9;">waiting: []</div>
            <div class="preempt-queue-item" style="background: #BBDEFB;">running: [A, B, C]</div>
        </div>
        
        <div class="preempt-queue">
            <div class="preempt-queue-title">抢占后：</div>
            <div class="preempt-queue-item" style="background: #FFCDD2;">waiting: [C]</div>
            <div class="preempt-queue-item" style="background: #BBDEFB;">running: [A, B]</div>
        </div>
    </div>
</body>
</html>
```

### 7.4 恢复时的 Prefix Caching

被抢占的 Sequence 在恢复时会经历完整的 `allocate` 流程：

```python
# 下一次调度时
if self.block_manager.can_allocate(seq_c):
    self.block_manager.allocate(seq_c)  # 可能命中自己之前的缓存!
```

**有趣的情况**：
- Seq C 被抢占前已生成部分 token
- 其 Block 被释放，但 `hash_to_block_id` 映射可能保留（如果没被覆写）
- 恢复时 `allocate` 可能命中这些缓存，减少重新计算的代价

---

## 八、配置与调优

### 8.1 关键配置参数

```python
@dataclass
class Config:
    model: str                              # 模型路径
    max_num_batched_tokens: int = 16384     # 单次迭代最大 token 数
    max_num_seqs: int = 512                 # 最大并发序列数
    max_model_len: int = 4096               # 最大序列长度
    gpu_memory_utilization: float = 0.9     # GPU 显存使用比例
    tensor_parallel_size: int = 1           # 张量并行度
    enforce_eager: bool = False             # 禁用 CUDA Graph
    kvcache_block_size: int = 256           # Block 大小（token 数）
    num_kvcache_blocks: int = -1            # Block 数量（自动计算）
```

**`kvcache_block_size = 256` 的选择理由**：
- 与 CUDA warp size（32）倍数对齐，利于 GPU 计算效率
- 足够大以减少 Block 管理开销和 hash 计算频率
- 256 tokens × 多层 × 多头的矩阵尺寸适合 Tensor Core

### 8.2 Block 数量估算

```python
# 单个 Block 的显存占用
block_bytes = (2 *                              # K + V
               hf_config.num_hidden_layers *    # 层数
               block_size *                     # 256
               num_kv_heads *                   # KV 头数
               hf_config.head_dim *             # 头维度
               hf_config.torch_dtype.itemsize)  # 数据类型大小

# 可用显存
available = total_gpu_memory * gpu_memory_utilization - model_weights - activation_peak

# Block 数量
num_kvcache_blocks = available // block_bytes
```

**Qwen3-0.6B 估算示例**：
- 参数：28 层，8 KV 头，128 维，bf16
- block_bytes = 2 × 28 × 256 × 8 × 128 × 2 = **29.36 MB**
- 假设可用显存 4.6 GB → num_blocks ≈ **160**
- 可支持的最大 token 数：160 × 256 = **40,960 tokens**

### 8.3 block_size 选择的权衡

| block_size | 优点 | 缺点 |
|------------|------|------|
| **大（256）** | Block 管理开销低<br>hash 计算频率低<br>flash_attn 效率高 | 内部碎片大（最后一个 Block 浪费）<br>Prefix Cache 粒度粗 |
| **小（16/32）** | 内部碎片小<br>Prefix Cache 细粒度<br>更灵活的内存管理 | 管理开销高<br>hash 计算频繁<br>block_table 更长 |

nano-vllm 固定使用 256，适合长序列、大 batch 场景。对于短对话、多轮交互场景，较小的 block_size 可能更优。

---

## 九、总结与思考

### 9.1 核心要点回顾

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .summary-container {
            background-color: white;
            padding: 30px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .summary-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 25px;
        }
        .summary-section {
            margin-bottom: 25px;
        }
        .summary-section-title {
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 2px solid #2196F3;
            display: inline-block;
        }
        .summary-points {
            padding-left: 20px;
        }
        .summary-point {
            margin-bottom: 8px;
            font-size: 13px;
            line-height: 1.6;
        }
        .summary-highlight {
            background: #E3F2FD;
            padding: 2px 6px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="summary-container">
        <div class="summary-title">核心要点回顾</div>
        
        <div class="summary-section">
            <div class="summary-section-title">Block / BlockManager / Sequence 三者关系</div>
            <div class="summary-points">
                <div class="summary-point">• <span class="summary-highlight">Block</span> 是物理存储单元，由 block_id 唯一标识，存储固定数量 token 的 KV Cache</div>
                <div class="summary-point">• <span class="summary-highlight">BlockManager</span> 管理 Block 的生命周期：分配、追加、释放、缓存查找</div>
                <div class="summary-point">• <span class="summary-highlight">Sequence</span> 持有 block_table，记录逻辑 Block 到物理 Block 的映射</div>
            </div>
        </div>
        
        <div class="summary-section">
            <div class="summary-section-title">Prefill 与 Decode 的关键差异</div>
            <div class="summary-points">
                <div class="summary-point">• Prefill：一次性处理整个 prompt，使用 <span class="summary-highlight">allocate</span> 分配多个 Block</div>
                <div class="summary-point">• Decode：逐 token 生成，使用 <span class="summary-highlight">may_append</span> 按需追加 Block</div>
                <div class="summary-point">• 资源不足时：Prefill 等待，Decode 抢占</div>
            </div>
        </div>
        
        <div class="summary-section">
            <div class="summary-section-title">Prefix Caching 的实现方式</div>
            <div class="summary-points">
                <div class="summary-point">• 使用 <span class="summary-highlight">链式 xxhash</span> 标识 Block 内容，确保前缀完全相同才能匹配</div>
                <div class="summary-point">• <span class="summary-highlight">ref_count</span> 实现 Block 共享，多个 Sequence 可复用同一 Block</div>
                <div class="summary-point">• 只有填满的 Block 才参与缓存，未填满的 Block hash = -1</div>
            </div>
        </div>
    </div>
</body>
</html>
```

### 9.2 nano-vllm vs vLLM

| 特性 | nano-vllm | vLLM |
|------|-----------|------|
| 代码量 | ~1,200 行 | ~100,000+ 行 |
| Copy-on-Write | ❌ 不支持 | ✅ 支持 |
| Beam Search Block 分叉 | ❌ 不支持 | ✅ 支持 |
| block_size | 固定 256 | 可配置 |
| 抢占策略 | 简单 LIFO | 多种策略可选 |
| Prefix Caching | 基础实现 | 更完善的 LRU 淘汰 |
| 分布式支持 | 基础 TP | 完整 TP/PP |

**nano-vllm 的定位**：教学和原型验证，通过简化设计展示 PagedAttention 的核心思想。

### 9.3 思考题

**Q1：如果两个请求的 prompt 有部分重叠但不完全相同，Prefix Caching 能生效吗？**

**答案**：只有**前缀完全相同的完整 Block** 才能共享。

例如：
- Seq1: "Hello world, how are you?" (假设占 1.5 个 Block)
- Seq2: "Hello world, what's up?" (同样 1.5 个 Block)

如果第一个 Block 内容完全相同，可以共享 Block 0。但由于后续内容不同，Block 1 的 hash 会因为链式传递而不同（即使 Block 1 的部分内容相同），无法共享。

**Q2：ref_count > 1 的 Block 如何处理写入？**

**答案**：在 nano-vllm 中，**共享 Block 只包含已填满的历史 token**，新 token 总是写入最后一个 Block（该 Block 的 hash = -1，不参与共享）。因此不会出现写入共享 Block 的情况。

如果需要支持 Beam Search 等场景（多个候选共享前缀，然后分叉），则需要实现 **Copy-on-Write**：当要写入 ref_count > 1 的 Block 时，先复制一份独立的 Block。

### 9.4 下期预告

**主题三：Scheduler 的 Continuous Batching 与迭代级调度**
- 为什么需要 Iteration-level 调度
- Prefill 和 Decode 的优先级权衡
- CUDA Graph 对调度的影响
- 吞吐量与延迟的 trade-off

---

## 附录：代码速查索引

| 功能点 | 文件 | 关键方法/类 |
|--------|------|-------------|
| Block 定义 | `engine/block_manager.py` | `Block` |
| Block 分配 | `engine/block_manager.py` | `BlockManager.allocate()` |
| Block 追加 | `engine/block_manager.py` | `BlockManager.may_append()` |
| Block 回收 | `engine/block_manager.py` | `BlockManager.deallocate()` |
| Prefix Cache hash | `engine/block_manager.py` | `BlockManager.compute_hash()` |
| Sequence 定义 | `engine/sequence.py` | `Sequence` |
| block_table | `engine/sequence.py` | `Sequence.block_table` |
| KV Cache 分配 | `engine/model_runner.py` | `ModelRunner.allocate_kv_cache()` |
| Prefill 准备 | `engine/model_runner.py` | `ModelRunner.prepare_prefill()` |
| Decode 准备 | `engine/model_runner.py` | `ModelRunner.prepare_decode()` |
| KV Cache 写入 | `layers/attention.py` | `store_kvcache_kernel()` |
| Attention 计算 | `layers/attention.py` | `Attention.forward()` |
| 运行时上下文 | `utils/context.py` | `Context`, `set_context()`, `get_context()` |
| 调度逻辑 | `engine/scheduler.py` | `Scheduler.schedule()` |
| 抢占处理 | `engine/scheduler.py` | `Scheduler.preempt()` |
| 后处理 | `engine/scheduler.py` | `Scheduler.postprocess()` |
| 配置参数 | `config.py` | `Config` |