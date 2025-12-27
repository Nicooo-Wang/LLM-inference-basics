# 主题一：推理全流程串讲（概览篇）

## 1. 课程目标

在正式开始之前，我想先说明一下这篇教程希望达成的目标：

- **理解 vLLM 的核心设计理念**：为什么要专门搞一个推理引擎，而不是直接拿 PyTorch 写个 generate 循环就完事？这背后涉及到传统方法遇到的实实在在的性能瓶颈，以及 vLLM 如何用一套受操作系统启发的设计来解决这些问题。

- **理清推理流程的全生命周期**：一个 prompt 进来之后，到底经历了哪些步骤才变成最终输出？我们会从头到尾追踪一条请求在系统里的完整旅程。

- **熟悉 nano-vllm 的核心架构**：nano-vllm 是一个轻量级的 vLLM 实现，用大约 1200 行 Python 代码实现了可读性很强的代码库，同时保持了与 vLLM 相当的推理速度。我们会了解它的核心模块划分以及各模块之间的协作关系。

---

## 2. 核心内容

### 一、vLLM 解决的核心痛点

如果你之前用过 Hugging Face Transformers 做推理，可能会发现一个尴尬的现象：GPU 显存占用很高，但实际利用率却时高时低，吞吐量也上不去。这不是你的代码有问题，而是传统推理方式本身就存在几个根本性的局限。

#### 1.1 传统推理的三大痛点

```html
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>传统推理的三大痛点</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 {
            text-align: center;
            color: #fff;
            margin-bottom: 30px;
            font-size: clamp(1.2rem, 4vw, 1.6rem);
        }
        .pain-point {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .pain-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }
        .pain-number {
            background: linear-gradient(135deg, #f5576c, #f093fb);
            color: #fff;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            flex-shrink: 0;
        }
        .pain-title { color: #f5576c; font-size: 1.1rem; font-weight: 600; }
        .pain-desc { color: #aaa; font-size: 0.9rem; line-height: 1.6; margin-bottom: 20px; }
        
        /* 痛点1：显存碎片化 */
        .memory-diagram {
            display: flex;
            flex-direction: column;
            gap: 12px;
            padding: 16px;
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
        }
        .memory-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .memory-label {
            color: #888;
            font-size: 0.75rem;
            width: 70px;
            flex-shrink: 0;
        }
        .memory-bar {
            flex: 1;
            height: 36px;
            background: #1a1a2e;
            border-radius: 6px;
            display: flex;
            overflow: hidden;
            border: 1px solid #333;
        }
        .mem-block {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.65rem;
            color: #fff;
            border-right: 1px solid rgba(0,0,0,0.3);
        }
        .mem-used-1 { background: #4facfe; }
        .mem-used-2 { background: #43e97b; }
        .mem-used-3 { background: #f093fb; }
        .mem-free { background: #2d3748; color: #666; }
        .mem-wasted { background: repeating-linear-gradient(45deg, #4a3728, #4a3728 5px, #3d2e1e 5px, #3d2e1e 10px); }
        .memory-legend {
            display: flex;
            gap: 16px;
            margin-top: 12px;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.75rem;
            color: #888;
        }
        .legend-box {
            width: 14px;
            height: 14px;
            border-radius: 3px;
        }
        .problem-callout {
            margin-top: 12px;
            padding: 12px;
            background: rgba(245, 87, 108, 0.15);
            border-left: 3px solid #f5576c;
            border-radius: 0 8px 8px 0;
            color: #f5576c;
            font-size: 0.85rem;
        }

        /* 痛点2：静态批处理 */
        .batch-diagram {
            padding: 16px;
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
        }
        .batch-timeline {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .batch-row {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .batch-label {
            color: #888;
            font-size: 0.75rem;
            width: 55px;
            flex-shrink: 0;
        }
        .batch-bar {
            flex: 1;
            height: 28px;
            display: flex;
            border-radius: 4px;
            overflow: hidden;
        }
        .batch-segment {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.65rem;
            color: #fff;
        }
        .batch-active { background: #4facfe; }
        .batch-waiting { background: repeating-linear-gradient(45deg, #4a5568, #4a5568 4px, #2d3748 4px, #2d3748 8px); color: #888; }
        .batch-idle { background: #2d3748; color: #666; }
        .time-arrow {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding: 8px 0;
            color: #666;
            font-size: 0.75rem;
        }
        .time-arrow::after {
            content: '';
            flex: 1;
            height: 2px;
            background: linear-gradient(90deg, transparent, #666);
            margin-left: 8px;
            margin-right: 4px;
        }

        /* 痛点3：KV Cache预分配 */
        .kv-diagram {
            padding: 16px;
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
        }
        .kv-requests {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .kv-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .kv-label {
            color: #888;
            font-size: 0.75rem;
            width: 60px;
            flex-shrink: 0;
        }
        .kv-bar-container {
            flex: 1;
            position: relative;
        }
        .kv-bar-bg {
            height: 32px;
            background: rgba(79, 172, 254, 0.2);
            border-radius: 6px;
            border: 1px dashed #4facfe;
            position: relative;
        }
        .kv-bar-actual {
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            background: #4facfe;
            border-radius: 6px 0 0 6px;
            display: flex;
            align-items: center;
            padding-left: 8px;
            font-size: 0.7rem;
            color: #fff;
        }
        .kv-stats {
            display: flex;
            gap: 20px;
            margin-top: 16px;
            padding-top: 12px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        .kv-stat {
            text-align: center;
        }
        .kv-stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #f5576c;
        }
        .kv-stat-label {
            font-size: 0.75rem;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 传统推理的三大痛点</h1>
        
        <!-- 痛点1：显存碎片化 -->
        <div class="pain-point">
            <div class="pain-header">
                <span class="pain-number">1</span>
                <span class="pain-title">显存碎片化 (Memory Fragmentation)</span>
            </div>
            <div class="pain-desc">
                传统方法为每个请求分配连续的显存空间存储 KV Cache。当请求完成释放内存后，留下的"空洞"往往无法被新请求完整利用，就像停车场里零散的空位停不下大车一样。
            </div>
            <div class="memory-diagram">
                <div class="memory-row">
                    <span class="memory-label">初始状态</span>
                    <div class="memory-bar">
                        <div class="mem-block mem-used-1" style="width:30%">Req A</div>
                        <div class="mem-block mem-used-2" style="width:25%">Req B</div>
                        <div class="mem-block mem-used-3" style="width:20%">Req C</div>
                        <div class="mem-block mem-free" style="width:25%">空闲</div>
                    </div>
                </div>
                <div class="memory-row">
                    <span class="memory-label">B 完成后</span>
                    <div class="memory-bar">
                        <div class="mem-block mem-used-1" style="width:30%">Req A</div>
                        <div class="mem-block mem-free" style="width:25%">空洞!</div>
                        <div class="mem-block mem-used-3" style="width:20%">Req C</div>
                        <div class="mem-block mem-free" style="width:25%">空闲</div>
                    </div>
                </div>
                <div class="memory-row">
                    <span class="memory-label">新请求 D</span>
                    <div class="memory-bar">
                        <div class="mem-block mem-used-1" style="width:30%">Req A</div>
                        <div class="mem-block mem-free" style="width:25%">无法使用</div>
                        <div class="mem-block mem-used-3" style="width:20%">Req C</div>
                        <div class="mem-block mem-wasted" style="width:25%">D需要35%</div>
                    </div>
                </div>
                <div class="memory-legend">
                    <div class="legend-item"><div class="legend-box" style="background:#4facfe"></div>已使用</div>
                    <div class="legend-item"><div class="legend-box" style="background:#2d3748"></div>空闲碎片</div>
                    <div class="legend-item"><div class="legend-box" style="background:repeating-linear-gradient(45deg, #4a3728, #4a3728 3px, #3d2e1e 3px, #3d2e1e 6px)"></div>无法分配</div>
                </div>
                <div class="problem-callout">
                    ⚠️ 新请求 D 需要 35% 连续空间，但现有空闲区都不够大，只能等待！
                </div>
            </div>
        </div>

        <!-- 痛点2：静态批处理 -->
        <div class="pain-point">
            <div class="pain-header">
                <span class="pain-number">2</span>
                <span class="pain-title">静态批处理的低效 (Static Batching)</span>
            </div>
            <div class="pain-desc">
                传统批处理需要等待整个 batch 中所有请求都完成后，才能开始处理下一批。先完成的请求只能"干等"，导致 GPU 利用率大幅波动。
            </div>
            <div class="batch-diagram">
                <div class="batch-timeline">
                    <div class="batch-row">
                        <span class="batch-label">请求 A</span>
                        <div class="batch-bar">
                            <div class="batch-segment batch-active" style="width:30%">生成中</div>
                            <div class="batch-segment batch-waiting" style="width:50%">等待其他请求...</div>
                            <div class="batch-segment batch-idle" style="width:20%"></div>
                        </div>
                    </div>
                    <div class="batch-row">
                        <span class="batch-label">请求 B</span>
                        <div class="batch-bar">
                            <div class="batch-segment batch-active" style="width:55%">生成中...</div>
                            <div class="batch-segment batch-waiting" style="width:25%">等待</div>
                            <div class="batch-segment batch-idle" style="width:20%"></div>
                        </div>
                    </div>
                    <div class="batch-row">
                        <span class="batch-label">请求 C</span>
                        <div class="batch-bar">
                            <div class="batch-segment batch-active" style="width:80%">生成很长的输出...</div>
                            <div class="batch-segment batch-idle" style="width:20%"></div>
                        </div>
                    </div>
                    <div class="batch-row">
                        <span class="batch-label">GPU</span>
                        <div class="batch-bar">
                            <div class="batch-segment" style="width:30%; background:#43e97b">满载</div>
                            <div class="batch-segment" style="width:25%; background:#f0b860">2/3</div>
                            <div class="batch-segment" style="width:25%; background:#f5576c">1/3</div>
                            <div class="batch-segment batch-idle" style="width:20%">空闲</div>
                        </div>
                    </div>
                </div>
                <div class="time-arrow">时间 →</div>
                <div class="problem-callout">
                    ⚠️ 请求 A 在 30% 时就完成了，但必须等到 C 完成（80%）才能释放资源！
                </div>
            </div>
        </div>

        <!-- 痛点3：KV Cache预分配 -->
        <div class="pain-point">
            <div class="pain-header">
                <span class="pain-number">3</span>
                <span class="pain-title">KV Cache 预分配浪费 (Over-Reservation)</span>
            </div>
            <div class="pain-desc">
                因为无法预知输出长度，系统只能按最大可能长度（如 2048 tokens）预分配 KV Cache。实际使用量往往远小于预分配量，造成大量显存浪费。
            </div>
            <div class="kv-diagram">
                <div class="kv-requests">
                    <div class="kv-row">
                        <span class="kv-label">请求 A</span>
                        <div class="kv-bar-container">
                            <div class="kv-bar-bg">
                                <div class="kv-bar-actual" style="width:15%">实际: 156 tokens</div>
                            </div>
                        </div>
                    </div>
                    <div class="kv-row">
                        <span class="kv-label">请求 B</span>
                        <div class="kv-bar-container">
                            <div class="kv-bar-bg">
                                <div class="kv-bar-actual" style="width:35%">实际: 412 tokens</div>
                            </div>
                        </div>
                    </div>
                    <div class="kv-row">
                        <span class="kv-label">请求 C</span>
                        <div class="kv-bar-container">
                            <div class="kv-bar-bg">
                                <div class="kv-bar-actual" style="width:8%">实际: 89 tokens</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div style="text-align:center; color:#666; font-size:0.75rem; margin-top:8px;">
                    虚线框 = 预分配 2048 tokens &nbsp;|&nbsp; 实心 = 实际使用
                </div>
                <div class="kv-stats">
                    <div class="kv-stat">
                        <div class="kv-stat-value">6144</div>
                        <div class="kv-stat-label">预分配 tokens</div>
                    </div>
                    <div class="kv-stat">
                        <div class="kv-stat-value">657</div>
                        <div class="kv-stat-label">实际使用</div>
                    </div>
                    <div class="kv-stat">
                        <div class="kv-stat-value">89%</div>
                        <div class="kv-stat-label">浪费率</div>
                    </div>
                </div>
                <div class="problem-callout">
                    ⚠️ 研究显示，现有系统因碎片和过度预留，浪费了 60%-80% 的显存！
                </div>
            </div>
        </div>
    </div>
</body>
</html>
```

**痛点一：KV Cache 的显存碎片化**

在自回归解码过程中，每个 token 都会生成对应的 Key 和 Value 向量，这些缓存需要保存在 GPU 显存中。传统方法为每个请求分配一块连续的内存空间。

问题在于，当某些请求完成并释放内存后，会在显存中留下不连续的"空洞"。新来的请求如果需要比这些空洞更大的连续空间，就无法利用这些已释放的内存——这与早期操作系统面临的内存碎片化问题如出一辙。

**痛点二：静态批处理的低效**

传统的批处理方式是"静态"的：凑齐一批请求，等整个 batch 全部完成，再接收下一批。问题是，不同请求的输出长度差异可能非常大——有的请求生成 10 个 token 就结束了，有的要生成 500 个。

这导致了一个荒谬的局面：先完成的请求不得不"陪跑"，占着显存干等其他请求。GPU 利用率随着请求逐个完成而不断下降，直到整个 batch 结束后才能恢复。

**痛点三：预分配造成的浪费**

因为系统无法预知一个请求最终会生成多少 token，只能按最大可能长度预分配 KV Cache。一个请求哪怕只生成了 50 个 token，也可能占着 2048 个 token 的显存空间。

研究发现，现有系统因为内存碎片和过度预留，浪费了 60% - 80% 的显存。这意味着你花了 80GB 显存的钱，实际只用上了 16-32GB。

---

#### 1.2 vLLM 的突破性创新

针对这些痛点，vLLM 提出了两个核心创新：PagedAttention 和连续批处理。

```html
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vLLM 的突破性创新</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 950px; margin: 0 auto; }
        h1 {
            text-align: center;
            color: #fff;
            margin-bottom: 30px;
            font-size: clamp(1.2rem, 4vw, 1.6rem);
        }
        .innovation {
            background: rgba(255,255,255,0.03);
            border-radius: 20px;
            padding: 28px;
            margin-bottom: 28px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .innovation-header {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 16px;
        }
        .innovation-icon {
            font-size: 2rem;
        }
        .innovation-title {
            color: #4ade80;
            font-size: 1.2rem;
            font-weight: 600;
        }
        .innovation-subtitle {
            color: #64748b;
            font-size: 0.85rem;
        }
        .innovation-desc {
            color: #94a3b8;
            font-size: 0.9rem;
            line-height: 1.7;
            margin-bottom: 24px;
        }

        /* PagedAttention 图表 */
        .paged-diagram {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 20px;
            align-items: start;
            padding: 20px;
            background: rgba(0,0,0,0.3);
            border-radius: 16px;
        }
        @media (max-width: 700px) {
            .paged-diagram {
                grid-template-columns: 1fr;
                gap: 16px;
            }
            .mapping-arrow { transform: rotate(90deg); }
        }
        .block-section {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .section-title {
            color: #64748b;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
            text-align: center;
        }
        .logical-blocks, .physical-blocks {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .sequence-row {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .seq-label {
            color: #94a3b8;
            font-size: 0.7rem;
            width: 45px;
            flex-shrink: 0;
        }
        .blocks-container {
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
        }
        .block {
            width: 36px;
            height: 36px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            font-weight: 600;
            color: #fff;
        }
        .block-a { background: linear-gradient(135deg, #3b82f6, #1d4ed8); }
        .block-b { background: linear-gradient(135deg, #10b981, #047857); }
        .block-c { background: linear-gradient(135deg, #f59e0b, #d97706); }
        .block-free { background: #334155; color: #64748b; border: 1px dashed #475569; }
        .mapping-arrow {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 8px;
            color: #4ade80;
            padding: 20px 0;
        }
        .arrow-text {
            font-size: 0.7rem;
            color: #64748b;
            writing-mode: vertical-rl;
            text-orientation: mixed;
        }
        .physical-section .blocks-container {
            display: grid;
            grid-template-columns: repeat(4, 36px);
            gap: 4px;
        }
        .block-table {
            margin-top: 20px;
            padding: 16px;
            background: rgba(74, 222, 128, 0.1);
            border-radius: 12px;
            border: 1px solid rgba(74, 222, 128, 0.3);
        }
        .block-table-title {
            color: #4ade80;
            font-size: 0.8rem;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .table-content {
            font-family: monospace;
            font-size: 0.75rem;
            color: #94a3b8;
            line-height: 1.8;
        }
        .table-row {
            display: flex;
            gap: 8px;
        }
        .table-key { color: #64748b; }
        .table-value { color: #4ade80; }

        /* 连续批处理图表 */
        .continuous-diagram {
            padding: 20px;
            background: rgba(0,0,0,0.3);
            border-radius: 16px;
        }
        .timeline-header {
            display: grid;
            grid-template-columns: 60px 1fr;
            gap: 12px;
            margin-bottom: 16px;
        }
        .time-labels {
            display: flex;
            justify-content: space-between;
            color: #64748b;
            font-size: 0.7rem;
        }
        .timeline-row {
            display: grid;
            grid-template-columns: 60px 1fr;
            gap: 12px;
            margin-bottom: 8px;
            align-items: center;
        }
        .row-label {
            color: #94a3b8;
            font-size: 0.75rem;
        }
        .timeline-bar {
            height: 32px;
            display: flex;
            border-radius: 6px;
            overflow: hidden;
            position: relative;
        }
        .timeline-segment {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.65rem;
            color: #fff;
            position: relative;
        }
        .seg-prefill { background: #8b5cf6; }
        .seg-decode { background: #3b82f6; }
        .seg-done { background: #334155; }
        .seg-empty { background: transparent; }
        .insert-marker {
            position: absolute;
            top: -20px;
            font-size: 0.65rem;
            color: #4ade80;
            white-space: nowrap;
        }
        .gpu-utilization {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        .gpu-bar {
            height: 24px;
            background: #1e293b;
            border-radius: 12px;
            overflow: hidden;
            display: flex;
        }
        .gpu-fill {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            color: #fff;
            transition: width 0.3s;
        }
        .gpu-high { background: linear-gradient(90deg, #4ade80, #22c55e); }
        .comparison-box {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 20px;
        }
        @media (max-width: 500px) {
            .comparison-box { grid-template-columns: 1fr; }
        }
        .compare-item {
            padding: 16px;
            border-radius: 12px;
            text-align: center;
        }
        .compare-bad {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        .compare-good {
            background: rgba(74, 222, 128, 0.15);
            border: 1px solid rgba(74, 222, 128, 0.3);
        }
        .compare-label {
            font-size: 0.75rem;
            color: #64748b;
            margin-bottom: 8px;
        }
        .compare-value {
            font-size: 1.4rem;
            font-weight: bold;
        }
        .compare-bad .compare-value { color: #ef4444; }
        .compare-good .compare-value { color: #4ade80; }
        .success-callout {
            margin-top: 16px;
            padding: 14px;
            background: rgba(74, 222, 128, 0.15);
            border-left: 3px solid #4ade80;
            border-radius: 0 8px 8px 0;
            color: #4ade80;
            font-size: 0.85rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 vLLM 的突破性创新</h1>

        <!-- 创新1：PagedAttention -->
        <div class="innovation">
            <div class="innovation-header">
                <span class="innovation-icon">📄</span>
                <div>
                    <div class="innovation-title">PagedAttention：分页内存管理</div>
                    <div class="innovation-subtitle">受操作系统虚拟内存启发的 KV Cache 管理</div>
                </div>
            </div>
            <div class="innovation-desc">
                核心思想：不再为每个请求分配连续内存，而是将 KV Cache 分割成固定大小的"块"（类似内存页）。通过 Block Table 维护逻辑块到物理块的映射，允许物理块在显存中非连续存放。
            </div>
            <div class="paged-diagram">
                <div class="block-section">
                    <div class="section-title">逻辑块（每个序列的视图）</div>
                    <div class="logical-blocks">
                        <div class="sequence-row">
                            <span class="seq-label">Seq A:</span>
                            <div class="blocks-container">
                                <div class="block block-a">L0</div>
                                <div class="block block-a">L1</div>
                                <div class="block block-a">L2</div>
                            </div>
                        </div>
                        <div class="sequence-row">
                            <span class="seq-label">Seq B:</span>
                            <div class="blocks-container">
                                <div class="block block-b">L0</div>
                                <div class="block block-b">L1</div>
                            </div>
                        </div>
                        <div class="sequence-row">
                            <span class="seq-label">Seq C:</span>
                            <div class="blocks-container">
                                <div class="block block-c">L0</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="mapping-arrow">
                    <span>→</span>
                    <span class="arrow-text">Block Table 映射</span>
                    <span>→</span>
                </div>
                
                <div class="block-section physical-section">
                    <div class="section-title">物理块（GPU 显存实际布局）</div>
                    <div class="blocks-container">
                        <div class="block block-a">P0</div>
                        <div class="block block-b">P1</div>
                        <div class="block block-c">P2</div>
                        <div class="block block-a">P3</div>
                        <div class="block block-free">空</div>
                        <div class="block block-b">P5</div>
                        <div class="block block-a">P6</div>
                        <div class="block block-free">空</div>
                    </div>
                </div>
            </div>
            
            <div class="block-table">
                <div class="block-table-title">📋 Block Table（逻辑 → 物理映射）</div>
                <div class="table-content">
                    <div class="table-row">
                        <span class="table-key">Seq A:</span>
                        <span class="table-value">[L0→P0, L1→P3, L2→P6]</span>
                    </div>
                    <div class="table-row">
                        <span class="table-key">Seq B:</span>
                        <span class="table-value">[L0→P1, L1→P5]</span>
                    </div>
                    <div class="table-row">
                        <span class="table-key">Seq C:</span>
                        <span class="table-value">[L0→P2]</span>
                    </div>
                </div>
            </div>
            
            <div class="comparison-box">
                <div class="compare-item compare-bad">
                    <div class="compare-label">传统方法内存浪费</div>
                    <div class="compare-value">60-80%</div>
                </div>
                <div class="compare-item compare-good">
                    <div class="compare-label">PagedAttention 内存浪费</div>
                    <div class="compare-value">&lt; 4%</div>
                </div>
            </div>
            
            <div class="success-callout">
                ✅ 物理块非连续存放，彻底解决内存碎片问题！浪费仅发生在每个序列的最后一个块。
            </div>
        </div>

        <!-- 创新2：连续批处理 -->
        <div class="innovation">
            <div class="innovation-header">
                <span class="innovation-icon">🔄</span>
                <div>
                    <div class="innovation-title">Continuous Batching：连续批处理</div>
                    <div class="innovation-subtitle">动态调度，最大化 GPU 利用率</div>
                </div>
            </div>
            <div class="innovation-desc">
                核心思想：不再等待整个 batch 完成，而是在每个迭代步骤动态地移除已完成的请求、插入新请求。GPU 始终保持满载运行，吞吐量大幅提升。
            </div>
            <div class="continuous-diagram">
                <div class="timeline-header">
                    <div></div>
                    <div class="time-labels">
                        <span>Step 1</span>
                        <span>Step 2</span>
                        <span>Step 3</span>
                        <span>Step 4</span>
                        <span>Step 5</span>
                    </div>
                </div>
                <div class="timeline-row">
                    <span class="row-label">请求 A</span>
                    <div class="timeline-bar">
                        <div class="timeline-segment seg-prefill" style="width:20%">Prefill</div>
                        <div class="timeline-segment seg-decode" style="width:30%">Decode</div>
                        <div class="timeline-segment seg-done" style="width:50%">✓ 完成</div>
                    </div>
                </div>
                <div class="timeline-row">
                    <span class="row-label">请求 B</span>
                    <div class="timeline-bar">
                        <div class="timeline-segment seg-prefill" style="width:20%">Prefill</div>
                        <div class="timeline-segment seg-decode" style="width:60%">Decode...</div>
                        <div class="timeline-segment seg-done" style="width:20%">✓</div>
                    </div>
                </div>
                <div class="timeline-row">
                    <span class="row-label">请求 C</span>
                    <div class="timeline-bar">
                        <div class="timeline-segment seg-empty" style="width:50%">
                            <span class="insert-marker">↓ A完成后立即插入</span>
                        </div>
                        <div class="timeline-segment seg-prefill" style="width:15%">Prefill</div>
                        <div class="timeline-segment seg-decode" style="width:35%">Decode</div>
                    </div>
                </div>
                <div class="timeline-row">
                    <span class="row-label">请求 D</span>
                    <div class="timeline-bar">
                        <div class="timeline-segment seg-empty" style="width:50%"></div>
                        <div class="timeline-segment seg-prefill" style="width:15%">
                            <span class="insert-marker">↓ 同时插入</span>
                        </div>
                        <div class="timeline-segment seg-decode" style="width:35%">Decode</div>
                    </div>
                </div>
                
                <div class="gpu-utilization">
                    <div class="row-label" style="margin-bottom:8px; color:#64748b; font-size:0.75rem;">GPU 利用率</div>
                    <div class="gpu-bar">
                        <div class="gpu-fill gpu-high" style="width:95%;">~95% 持续高利用率</div>
                    </div>
                </div>
            </div>
            
            <div class="success-callout">
                ✅ 请求 A 完成后，C 和 D 立即插入执行，GPU 始终保持满载！吞吐量相比传统方法提升最高达 24 倍。
            </div>
        </div>
    </div>
</body>
</html>
```

**创新一：PagedAttention（分页注意力）**

PagedAttention 的核心思想直接借鉴了操作系统的虚拟内存管理。在操作系统中，程序看到的是连续的虚拟地址空间，但实际的物理内存页可以分散存放在任何位置，通过页表来维护映射关系。

vLLM 将这个思想应用到 KV Cache 管理上：每个序列的 KV Cache 被分割成固定大小的"块"，每个块包含固定数量 token 的 Key 和 Value。这些物理块可以在 GPU 显存中非连续存放，通过 Block Table 维护逻辑块到物理块的映射。

这样一来，内存碎片问题就彻底解决了——因为所有块大小相同，任何空闲块都可以被任何需要新块的请求使用。浪费只发生在每个序列的最后一个块（因为最后一个块可能没有填满）。

**创新二：连续批处理（Continuous Batching）**

连续批处理打破了传统的"等待整个 batch 完成"的模式。在每个推理迭代步骤：首先检查是否有请求完成（遇到 EOS 或达到最大长度），将其移出运行队列并释放资源；然后检查等待队列中是否有新请求可以调度，如果有足够的资源就将其加入运行队列。

这样，GPU 可以始终保持接近满载的状态。一个请求完成后，它的资源立即被回收并分配给等待中的请求，不存在"陪跑"的浪费。

---

#### 1.3 nano-vllm 的教学定位

nano-vllm 是一个轻量级的 vLLM 实现，专为学习和理解 vLLM 核心思想而设计。它用约 1200 行干净的 Python 代码实现了 vLLM 的核心功能，同时保留了关键的性能优化组件。

nano-vllm 的特点包括：剥离了复杂的底层 C++/CUDA 代码，专注于核心调度算法的 Python 实现；使用 Flash Attention 和 Triton kernels 保证推理速度；支持 Continuous Batching、PagedAttention、前缀缓存、张量并行等核心特性。

简单来说，如果你想理解 vLLM 的核心思想而不是被底层优化细节淹没，nano-vllm 是一个理想的学习起点。

---

### 二、系统架构概览

下面这张架构图展示了 nano-vllm/vLLM 的核心类及其依赖关系。特别注意在多卡（Tensor Parallel）场景下，哪些类是单实例的，哪些是多实例的。

```html
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>nano-vLLM 架构图</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(180deg, #0a0a1a 0%, #1a1a3a 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        h1 {
            text-align: center;
            color: #fff;
            margin-bottom: 8px;
            font-size: clamp(1.2rem, 4vw, 1.6rem);
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 24px;
            font-size: 0.85rem;
        }
        
        .architecture {
            position: relative;
            padding: 20px;
        }
        
        /* 主控层 */
        .control-layer {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }
        .engine-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 20px 40px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            position: relative;
        }
        .engine-box::after {
            content: '';
            position: absolute;
            bottom: -20px;
            left: 50%;
            transform: translateX(-50%);
            width: 3px;
            height: 20px;
            background: #667eea;
        }
        .box-title {
            color: #fff;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .box-file {
            color: rgba(255,255,255,0.7);
            font-size: 0.75rem;
            font-family: monospace;
            margin-top: 4px;
        }
        .box-desc {
            color: rgba(255,255,255,0.8);
            font-size: 0.8rem;
            margin-top: 8px;
        }
        .instance-badge {
            position: absolute;
            top: -8px;
            right: -8px;
            background: #1a1a3a;
            color: #667eea;
            font-size: 0.65rem;
            padding: 3px 8px;
            border-radius: 10px;
            border: 2px solid #667eea;
        }
        
        /* 调度层 */
        .scheduler-layer {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .scheduler-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 14px;
            padding: 16px 28px;
            text-align: center;
            box-shadow: 0 6px 24px rgba(240, 147, 251, 0.25);
            position: relative;
        }
        .block-manager-box {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            box-shadow: 0 6px 24px rgba(79, 172, 254, 0.25);
        }
        
        /* 连接线 */
        .connector {
            display: flex;
            justify-content: center;
            margin: 0 0 20px 0;
            position: relative;
        }
        .connector-line {
            width: 60%;
            height: 30px;
            border: 3px solid #4a5568;
            border-top: none;
            border-radius: 0 0 20px 20px;
            position: relative;
        }
        .connector-line::before, .connector-line::after {
            content: '↓';
            position: absolute;
            bottom: -12px;
            color: #4a5568;
            font-size: 0.9rem;
        }
        .connector-line::before { left: -6px; }
        .connector-line::after { right: -6px; }
        
        /* 执行层 */
        .execution-layer {
            background: rgba(255,255,255,0.03);
            border-radius: 20px;
            padding: 24px;
            border: 2px dashed rgba(255,255,255,0.15);
            margin-top: 20px;
        }
        .layer-title {
            color: #888;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            text-align: center;
            margin-bottom: 16px;
        }
        .multi-instance-note {
            color: #43e97b;
            font-size: 0.7rem;
            text-align: center;
            margin-bottom: 12px;
        }
        .workers-container {
            display: flex;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
        }
        .worker-box {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            border-radius: 12px;
            padding: 14px 20px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(67, 233, 123, 0.2);
            position: relative;
            min-width: 140px;
        }
        .worker-box .box-title {
            font-size: 0.95rem;
        }
        .gpu-badge {
            position: absolute;
            top: -6px;
            right: -6px;
            background: #1a1a3a;
            color: #43e97b;
            font-size: 0.6rem;
            padding: 2px 6px;
            border-radius: 8px;
            border: 2px solid #43e97b;
        }
        
        /* 模型层 */
        .model-layer {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        .models-container {
            display: flex;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
        }
        .model-box {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border-radius: 10px;
            padding: 12px 18px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);
            position: relative;
            min-width: 130px;
        }
        .model-box .box-title { font-size: 0.9rem; }
        
        /* 数据结构层 */
        .data-layer {
            margin-top: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }
        .data-box {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 16px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .data-title {
            color: #f0b860;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .data-desc {
            color: #888;
            font-size: 0.8rem;
            line-height: 1.5;
        }
        .data-file {
            color: #666;
            font-size: 0.7rem;
            font-family: monospace;
            margin-top: 8px;
        }
        
        /* 图例 */
        .legend {
            margin-top: 24px;
            display: flex;
            justify-content: center;
            gap: 24px;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.75rem;
            color: #888;
        }
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .legend-single { background: #667eea; }
        .legend-multi { background: #43e97b; }
        
        /* 箭头说明 */
        .arrow-label {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            color: #666;
            font-size: 0.75rem;
            margin: 12px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏗️ nano-vLLM 系统架构</h1>
        <p class="subtitle">类依赖关系与多卡实例分布</p>
        
        <div class="architecture">
            <!-- 主控层 -->
            <div class="control-layer">
                <div class="engine-box">
                    <span class="instance-badge">单实例</span>
                    <div class="box-title">LLMEngine</div>
                    <div class="box-file">engine.py</div>
                    <div class="box-desc">系统总控制器，驱动 step() 循环</div>
                </div>
            </div>
            
            <div class="arrow-label">
                <span>↓</span> 持有并协调 <span>↓</span>
            </div>
            
            <!-- 调度层 -->
            <div class="scheduler-layer">
                <div class="scheduler-box">
                    <span class="instance-badge">单实例</span>
                    <div class="box-title">Scheduler</div>
                    <div class="box-file">scheduler.py</div>
                    <div class="box-desc">请求调度决策</div>
                </div>
                <div class="scheduler-box block-manager-box">
                    <span class="instance-badge">单实例</span>
                    <div class="box-title">BlockManager</div>
                    <div class="box-file">block_manager.py</div>
                    <div class="box-desc">KV Cache 内存管理</div>
                </div>
            </div>
            
            <div class="arrow-label">
                <span>↓</span> 调度指令下发 <span>↓</span>
            </div>
            
            <!-- 执行层 -->
            <div class="execution-layer">
                <div class="layer-title">执行层（Tensor Parallel 场景下多实例）</div>
                <div class="multi-instance-note">⚡ 以下组件在每张 GPU 上各有一个实例</div>
                
                <div class="workers-container">
                    <div class="worker-box">
                        <span class="gpu-badge">GPU 0</span>
                        <div class="box-title">Worker</div>
                        <div class="box-file">worker.py</div>
                    </div>
                    <div class="worker-box">
                        <span class="gpu-badge">GPU 1</span>
                        <div class="box-title">Worker</div>
                        <div class="box-file">worker.py</div>
                    </div>
                    <div class="worker-box" style="opacity:0.6;">
                        <span class="gpu-badge">GPU N</span>
                        <div class="box-title">Worker</div>
                        <div class="box-file">worker.py</div>
                    </div>
                </div>
                
                <div class="arrow-label" style="margin-top:16px;">
                    ↓ 每个 Worker 持有 ↓
                </div>
                
                <div class="model-layer">
                    <div class="models-container">
                        <div class="model-box">
                            <span class="gpu-badge">GPU 0</span>
                            <div class="box-title">Model (分片)</div>
                            <div class="box-file">model.py</div>
                        </div>
                        <div class="model-box">
                            <span class="gpu-badge">GPU 1</span>
                            <div class="box-title">Model (分片)</div>
                            <div class="box-file">model.py</div>
                        </div>
                        <div class="model-box" style="opacity:0.6;">
                            <span class="gpu-badge">GPU N</span>
                            <div class="box-title">Model (分片)</div>
                            <div class="box-file">model.py</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 核心数据结构 -->
            <div class="data-layer">
                <div class="data-box">
                    <div class="data-title">SequenceGroup</div>
                    <div class="data-desc">封装一个请求的完整信息：prompt tokens、生成的 tokens、采样参数、状态等</div>
                    <div class="data-file">sequence.py</div>
                </div>
                <div class="data-box">
                    <div class="data-title">BlockTable</div>
                    <div class="data-desc">维护逻辑块到物理块的映射，是 PagedAttention 的核心数据结构</div>
                    <div class="data-file">block_manager.py</div>
                </div>
                <div class="data-box">
                    <div class="data-title">KVCache</div>
                    <div class="data-desc">预分配的 GPU 显存张量，存储所有序列的 Key/Value 缓存</div>
                    <div class="data-file">worker.py</div>
                </div>
                <div class="data-box">
                    <div class="data-title">SamplingParams</div>
                    <div class="data-desc">采样配置：温度、top_p、top_k、max_tokens 等</div>
                    <div class="data-file">sampling_params.py</div>
                </div>
            </div>
            
            <!-- 图例 -->
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-dot legend-single"></div>
                    <span>单实例（全局唯一）</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot legend-multi"></div>
                    <span>多实例（每 GPU 一个）</span>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
```

让我详细解释各个组件及其在多卡场景下的实例分布：

**单实例组件（全局唯一）**

这些组件在整个系统中只有一个实例，负责全局的协调和管理工作：

- **LLMEngine**：系统的总控制器，持有 Scheduler 和 Worker 的引用。它的 `step()` 方法是整个推理循环的驱动核心。无论有多少张 GPU，只需要一个 Engine 来协调它们。

- **Scheduler**：全局的调度器，维护 Waiting、Running、Finished 三个队列。它需要对所有请求有全局视图，因此只能是单实例。

- **BlockManager**：管理 KV Cache 的物理块分配。虽然物理块分布在多张 GPU 上，但块的分配和映射决策是中心化的，因此 BlockManager 是单实例的。

**多实例组件（每 GPU 一个）**

这些组件在每张 GPU 上都有独立的实例：

- **Worker**：每张 GPU 上有一个 Worker 实例，负责在该 GPU 上执行计算。Worker 持有该 GPU 上的 KV Cache 张量和模型分片。

- **Model**：在张量并行（Tensor Parallel）场景下，模型的权重被切分到多张 GPU 上。每张 GPU 上有一个 Model 实例，持有权重的一个分片。

**核心数据结构**

- **SequenceGroup**：封装一个请求的完整状态，包括 prompt tokens、已生成的 tokens、采样参数、当前状态（Waiting/Running/Finished）等。

- **BlockTable**：每个 SequenceGroup 有自己的 BlockTable，记录逻辑块到物理块的映射。

- **KVCache**：预分配在 GPU 显存中的张量，按块组织，存储所有序列的 Key 和 Value。

- **SamplingParams**：采样配置参数，如温度、top_p、max_tokens 等。

---

### 三、一条请求的全流程拆解

现在让我们跟踪一条请求从输入到输出的完整生命周期，包括 Tokenization 和 Detokenization 的细节。

```html
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>请求完整生命周期</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(180deg, #0a0a1a 0%, #1a1a3a 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 {
            text-align: center;
            color: #fff;
            margin-bottom: 30px;
            font-size: clamp(1.2rem, 4vw, 1.6rem);
        }
        
        .lifecycle {
            position: relative;
        }
        
        .stage {
            display: grid;
            grid-template-columns: 80px 1fr;
            gap: 20px;
            margin-bottom: 0;
            position: relative;
        }
        
        .stage-indicator {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .stage-number {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            font-weight: bold;
            color: #fff;
            position: relative;
            z-index: 2;
        }
        .stage-line {
            width: 3px;
            flex: 1;
            min-height: 40px;
        }
        .stage:last-child .stage-line { display: none; }
        
        .stage-content {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 20px;
        }
        .stage-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
        }
        .stage-title {
            font-size: 1.05rem;
            font-weight: 600;
        }
        .stage-subtitle {
            color: #666;
            font-size: 0.8rem;
        }
        .stage-desc {
            color: #94a3b8;
            font-size: 0.88rem;
            line-height: 1.7;
            margin-bottom: 16px;
        }
        
        /* 颜色主题 */
        .stage-1 .stage-number { background: linear-gradient(135deg, #f59e0b, #d97706); }
        .stage-1 .stage-line { background: linear-gradient(180deg, #f59e0b, #8b5cf6); }
        .stage-1 .stage-title { color: #f59e0b; }
        
        .stage-2 .stage-number { background: linear-gradient(135deg, #8b5cf6, #6366f1); }
        .stage-2 .stage-line { background: linear-gradient(180deg, #8b5cf6, #ec4899); }
        .stage-2 .stage-title { color: #8b5cf6; }
        
        .stage-3 .stage-number { background: linear-gradient(135deg, #ec4899, #f43f5e); }
        .stage-3 .stage-line { background: linear-gradient(180deg, #ec4899, #3b82f6); }
        .stage-3 .stage-title { color: #ec4899; }
        
        .stage-4 .stage-number { background: linear-gradient(135deg, #3b82f6, #0ea5e9); }
        .stage-4 .stage-line { background: linear-gradient(180deg, #3b82f6, #10b981); }
        .stage-4 .stage-title { color: #3b82f6; }
        
        .stage-5 .stage-number { background: linear-gradient(135deg, #10b981, #059669); }
        .stage-5 .stage-line { background: linear-gradient(180deg, #10b981, #06b6d4); }
        .stage-5 .stage-title { color: #10b981; }
        
        .stage-6 .stage-number { background: linear-gradient(135deg, #06b6d4, #0891b2); }
        .stage-6 .stage-title { color: #06b6d4; }
        
        /* 流程图 */
        .flow-diagram {
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 16px;
        }
        .flow-row {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .flow-box {
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 0.75rem;
            color: #fff;
            text-align: center;
            min-width: 80px;
        }
        .flow-input { background: #374151; border: 1px solid #4b5563; }
        .flow-process { background: #4338ca; }
        .flow-output { background: #059669; }
        .flow-arrow { color: #4b5563; font-size: 1.2rem; }
        
        /* Token 展示 */
        .token-display {
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 16px;
            font-family: monospace;
        }
        .token-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }
        .token-label {
            color: #64748b;
            font-size: 0.75rem;
            width: 60px;
            flex-shrink: 0;
        }
        .token {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
        }
        .token-text { background: #374151; color: #e5e7eb; }
        .token-id { background: #1e3a5f; color: #60a5fa; }
        .token-special { background: #4c1d95; color: #c4b5fd; }
        
        /* 状态转换 */
        .state-flow {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
            padding: 16px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
        }
        .state-box {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .state-waiting { background: #fef3c7; color: #92400e; }
        .state-running { background: #dbeafe; color: #1e40af; }
        .state-finished { background: #d1fae5; color: #065f46; }
        .state-arrow { color: #6b7280; }
        
        /* 代码展示 */
        .code-block {
            background: #0d1117;
            border-radius: 10px;
            padding: 14px;
            font-family: 'Fira Code', monospace;
            font-size: 0.75rem;
            color: #c9d1d9;
            overflow-x: auto;
            border: 1px solid #30363d;
        }
        .code-comment { color: #8b949e; }
        .code-keyword { color: #ff7b72; }
        .code-string { color: #a5d6ff; }
        .code-func { color: #d2a8ff; }
        
        /* 详情列表 */
        .detail-list {
            margin-top: 12px;
        }
        .detail-item {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            margin-bottom: 8px;
            font-size: 0.85rem;
            color: #94a3b8;
        }
        .detail-icon { color: #6366f1; flex-shrink: 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔄 一条请求的完整生命周期</h1>
        
        <div class="lifecycle">
            <!-- 阶段1：Tokenization -->
            <div class="stage stage-1">
                <div class="stage-indicator">
                    <div class="stage-number">1</div>
                    <div class="stage-line"></div>
                </div>
                <div class="stage-content">
                    <div class="stage-header">
                        <span class="stage-title">Tokenization 分词</span>
                        <span class="stage-subtitle">文本 → Token IDs</span>
                    </div>
                    <div class="stage-desc">
                        用户输入的自然语言文本首先通过 Tokenizer 转换为模型可以理解的 Token ID 序列。Tokenizer 通常使用 BPE（Byte Pair Encoding）或 SentencePiece 算法。
                    </div>
                    <div class="token-display">
                        <div class="token-row">
                            <span class="token-label">输入:</span>
                            <span class="token token-text">"请解释什么是深度学习"</span>
                        </div>
                        <div class="token-row">
                            <span class="token-label">分词:</span>
                            <span class="token token-text">请</span>
                            <span class="token token-text">解释</span>
                            <span class="token token-text">什么</span>
                            <span class="token token-text">是</span>
                            <span class="token token-text">深度</span>
                            <span class="token token-text">学习</span>
                        </div>
                        <div class="token-row">
                            <span class="token-label">Token IDs:</span>
                            <span class="token token-id">2048</span>
                            <span class="token token-id">15234</span>
                            <span class="token token-id">8821</span>
                            <span class="token token-id">1025</span>
                            <span class="token token-id">29405</span>
                            <span class="token token-id">18562</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 阶段2：请求封装 -->
            <div class="stage stage-2">
                <div class="stage-indicator">
                    <div class="stage-number">2</div>
                    <div class="stage-line"></div>
                </div>
                <div class="stage-content">
                    <div class="stage-header">
                        <span class="stage-title">Request 封装</span>
                        <span class="stage-subtitle">创建 SequenceGroup</span>
                    </div>
                    <div class="stage-desc">
                        Token IDs 连同采样参数被封装成 SequenceGroup 对象，加入 Scheduler 的 Waiting 队列等待调度。
                    </div>
                    <div class="code-block">
<span class="code-comment"># 创建序列组</span>
seq_group = SequenceGroup(
    request_id=<span class="code-string">"req-001"</span>,
    prompt_token_ids=[<span class="code-string">2048, 15234, 8821, ...</span>],
    sampling_params=SamplingParams(
        temperature=<span class="code-string">0.7</span>,
        max_tokens=<span class="code-string">256</span>
    )
)
scheduler.waiting.append(seq_group)
                    </div>
                    <div class="state-flow" style="margin-top:16px;">
                        <span class="state-box state-waiting">Waiting</span>
                        <span class="state-arrow">← 初始状态</span>
                    </div>
                </div>
            </div>
            
            <!-- 阶段3：调度 -->
            <div class="stage stage-3">
                <div class="stage-indicator">
                    <div class="stage-number">3</div>
                    <div class="stage-line"></div>
                </div>
                <div class="stage-content">
                    <div class="stage-header">
                        <span class="stage-title">Scheduling 调度</span>
                        <span class="stage-subtitle">分配资源，进入执行</span>
                    </div>
                    <div class="stage-desc">
                        Scheduler 检查 GPU 显存容量，选择可执行的请求。BlockManager 为选中的请求分配物理块，请求状态变为 Running。
                    </div>
                    <div class="flow-diagram">
                        <div class="flow-row">
                            <div class="flow-box flow-input">Waiting 队列</div>
                            <span class="flow-arrow">→</span>
                            <div class="flow-box flow-process">Scheduler 检查容量</div>
                            <span class="flow-arrow">→</span>
                            <div class="flow-box flow-process">BlockManager 分配块</div>
                            <span class="flow-arrow">→</span>
                            <div class="flow-box flow-output">Running 队列</div>
                        </div>
                    </div>
                    <div class="state-flow" style="margin-top:16px;">
                        <span class="state-box state-waiting">Waiting</span>
                        <span class="state-arrow">→</span>
                        <span class="state-box state-running">Running</span>
                    </div>
                </div>
            </div>
            
            <!-- 阶段4：Prefill -->
            <div class="stage stage-4">
                <div class="stage-indicator">
                    <div class="stage-number">4</div>
                    <div class="stage-line"></div>
                </div>
                <div class="stage-content">
                    <div class="stage-header">
                        <span class="stage-title">Prefill 预填充</span>
                        <span class="stage-subtitle">处理完整 Prompt</span>
                    </div>
                    <div class="stage-desc">
                        整个 Prompt 序列一次性送入模型，计算所有 token 的 KV Cache。这是计算密集型阶段，涉及大量矩阵乘法。
                    </div>
                    <div class="detail-list">
                        <div class="detail-item">
                            <span class="detail-icon">▸</span>
                            <span>所有 prompt tokens 并行通过模型的 Attention 层</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-icon">▸</span>
                            <span>为每个 token 生成 Key 和 Value 向量</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-icon">▸</span>
                            <span>KV Cache 存入 BlockManager 管理的物理块</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-icon">▸</span>
                            <span>输出最后一个位置的 logits，采样得到第一个生成 token</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 阶段5：Decode -->
            <div class="stage stage-5">
                <div class="stage-indicator">
                    <div class="stage-number">5</div>
                    <div class="stage-line"></div>
                </div>
                <div class="stage-content">
                    <div class="stage-header">
                        <span class="stage-title">Decode 解码</span>
                        <span class="stage-subtitle">迭代生成 Token</span>
                    </div>
                    <div class="stage-desc">
                        进入循环生成模式。每次迭代只处理上一步生成的 token，通过 PagedAttention 读取 KV Cache，计算注意力并采样下一个 token。
                    </div>
                    <div class="token-display">
                        <div class="token-row">
                            <span class="token-label">Step 1:</span>
                            <span class="token token-text">[prompt]</span>
                            <span style="color:#4ade80;">→</span>
                            <span class="token token-special">深度</span>
                        </div>
                        <div class="token-row">
                            <span class="token-label">Step 2:</span>
                            <span class="token token-text">[prompt] 深度</span>
                            <span style="color:#4ade80;">→</span>
                            <span class="token token-special">学习</span>
                        </div>
                        <div class="token-row">
                            <span class="token-label">Step 3:</span>
                            <span class="token token-text">[prompt] 深度学习</span>
                            <span style="color:#4ade80;">→</span>
                            <span class="token token-special">是</span>
                        </div>
                        <div class="token-row">
                            <span class="token-label">...</span>
                            <span style="color:#64748b; font-size:0.8rem;">继续迭代直到遇到 EOS 或达到 max_tokens</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 阶段6：Detokenization & 结束 -->
            <div class="stage stage-6">
                <div class="stage-indicator">
                    <div class="stage-number">6</div>
                    <div class="stage-line"></div>
                </div>
                <div class="stage-content">
                    <div class="stage-header">
                        <span class="stage-title">Detokenization & 资源回收</span>
                        <span class="stage-subtitle">Token IDs → 文本</span>
                    </div>
                    <div class="stage-desc">
                        当遇到 EOS token 或达到最大长度时，请求完成。生成的 Token IDs 通过 Tokenizer 反向转换为文本，BlockManager 回收物理块。
                    </div>
                    <div class="token-display">
                        <div class="token-row">
                            <span class="token-label">Token IDs:</span>
                            <span class="token token-id">29405</span>
                            <span class="token token-id">18562</span>
                            <span class="token token-id">1025</span>
                            <span class="token token-id">...</span>
                            <span class="token token-id">2</span>
                        </div>
                        <div class="token-row">
                            <span class="token-label">输出:</span>
                            <span class="token token-text">"深度学习是一种机器学习方法..."</span>
                        </div>
                    </div>
                    <div class="state-flow" style="margin-top:16px;">
                        <span class="state-box state-running">Running</span>
                        <span class="state-arrow">→</span>
                        <span class="state-box state-finished">Finished</span>
                        <span class="state-arrow">→ 回收物理块</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
```

#### 关于 Tokenizer：文本与 Token 的桥梁

Tokenizer 是 LLM 推理流程中不可或缺的组件，它负责文本和 Token ID 之间的双向转换。

**Tokenization（分词）**

Tokenization 将自然语言文本转换为 Token ID 序列。现代 LLM 通常使用子词分词算法，如 BPE（Byte Pair Encoding）、WordPiece 或 SentencePiece。这些算法能够平衡词汇表大小和表示能力——常见词可能是一个 token，而罕见词则被拆分成多个子词 token。

例如，对于输入 "深度学习是人工智能的核心技术"，不同的分词结果可能是：按字分词得到单独的字符 token，或者按子词分词将 "深度"、"学习"、"人工智能" 等常见组合作为单个 token。

**Detokenization（反分词）**

Detokenization 是 Tokenization 的逆过程，将 Token ID 序列转换回人类可读的文本。需要注意的是，由于分词算法的特性（比如 BPE 的合并规则），detokenization 需要正确处理子词边界和特殊字符。

**nano-vllm 中的实现**

nano-vllm 使用 HuggingFace Transformers 的 Tokenizer。在代码中，tokenization 发生在请求进入系统时（`engine.py` 中的 `add_request` 方法），detokenization 发生在请求完成时或流式输出过程中。

```python
# Tokenization
token_ids = tokenizer.encode("你好，世界")  # [12043, 8834, 1922]

# Detokenization
text = tokenizer.decode([12043, 8834, 1922])  # "你好，世界"
```

---

### 四、Prefill 与 Decode 的差异化处理

Prefill 和 Decode 是 LLM 推理中两个计算特性截然不同的阶段。理解它们的差异对于优化推理性能至关重要。

```html
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prefill vs Decode</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 950px; margin: 0 auto; }
        h1 {
            text-align: center;
            color: #fff;
            margin-bottom: 30px;
            font-size: clamp(1.2rem, 4vw, 1.6rem);
        }
        
        .phase-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 30px;
        }
        @media (max-width: 750px) {
            .phase-container { grid-template-columns: 1fr; }
        }
        
        .phase-card {
            background: rgba(255,255,255,0.03);
            border-radius: 20px;
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .phase-header {
            text-align: center;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .phase-icon { font-size: 2.5rem; margin-bottom: 8px; }
        .phase-title { font-size: 1.2rem; font-weight: 600; margin-bottom: 4px; }
        .prefill-card .phase-title { color: #8b5cf6; }
        .decode-card .phase-title { color: #f59e0b; }
        .phase-subtitle { color: #64748b; font-size: 0.85rem; }
        
        /* 注意力矩阵可视化 */
        .attention-viz {
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .viz-title {
            color: #94a3b8;
            font-size: 0.75rem;
            text-align: center;
            margin-bottom: 12px;
        }
        .matrix-container {
            display: flex;
            justify-content: center;
            align-items: flex-start;
            gap: 8px;
        }
        .axis-label {
            color: #64748b;
            font-size: 0.65rem;
            writing-mode: vertical-rl;
            text-orientation: mixed;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .matrix-wrapper {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .matrix {
            display: grid;
            gap: 2px;
        }
        .prefill-matrix { grid-template-columns: repeat(6, 1fr); }
        .decode-matrix { grid-template-columns: repeat(7, 1fr); }
        .matrix-cell {
            width: 28px;
            height: 28px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.55rem;
            color: rgba(255,255,255,0.7);
        }
        .cell-active { background: linear-gradient(135deg, #8b5cf6, #6366f1); }
        .cell-compute { background: linear-gradient(135deg, #f59e0b, #ea580c); }
        .cell-cached { background: #334155; color: #64748b; }
        .cell-mask { background: #1e293b; color: #374151; }
        .cell-new { 
            background: linear-gradient(135deg, #10b981, #059669); 
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
        }
        
        .token-labels {
            display: flex;
            gap: 2px;
            margin-top: 6px;
        }
        .token-label {
            width: 28px;
            text-align: center;
            font-size: 0.55rem;
            color: #64748b;
        }
        .token-label-new { color: #10b981; font-weight: 600; }
        
        /* 计算量指示 */
        .compute-indicator {
            margin-top: 16px;
            padding: 12px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }
        .compute-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }
        .compute-label {
            color: #64748b;
            font-size: 0.75rem;
            width: 80px;
            flex-shrink: 0;
        }
        .compute-bar {
            flex: 1;
            height: 20px;
            background: #1e293b;
            border-radius: 10px;
            overflow: hidden;
        }
        .compute-fill {
            height: 100%;
            border-radius: 10px;
            display: flex;
            align-items: center;
            padding-left: 8px;
            font-size: 0.7rem;
            color: #fff;
        }
        .fill-compute { background: linear-gradient(90deg, #8b5cf6, #6366f1); }
        .fill-memory { background: linear-gradient(90deg, #f59e0b, #ea580c); }
        
        /* 特性列表 */
        .feature-list {
            margin-top: 16px;
        }
        .feature-item {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            margin-bottom: 10px;
            font-size: 0.85rem;
            color: #94a3b8;
        }
        .feature-icon { flex-shrink: 0; }
        .prefill-card .feature-icon { color: #8b5cf6; }
        .decode-card .feature-icon { color: #f59e0b; }
        
        /* 序列时间线 */
        .sequence-timeline {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .timeline-title {
            color: #fff;
            font-size: 1rem;
            margin-bottom: 20px;
            text-align: center;
        }
        .timeline-viz {
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 20px;
        }
        .seq-row {
            display: flex;
            align-items: center;
            margin-bottom: 16px;
        }
        .seq-label {
            color: #64748b;
            font-size: 0.75rem;
            width: 70px;
            flex-shrink: 0;
        }
        .seq-tokens {
            display: flex;
            gap: 3px;
            flex-wrap: wrap;
        }
        .seq-token {
            width: 32px;
            height: 32px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.65rem;
            color: #fff;
        }
        .tok-prompt { background: #334155; }
        .tok-prefill { background: linear-gradient(135deg, #8b5cf6, #6366f1); }
        .tok-decode { background: linear-gradient(135deg, #f59e0b, #ea580c); }
        .tok-current { 
            background: linear-gradient(135deg, #10b981, #059669);
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        }
        
        .phase-indicator {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        .phase-badge {
            padding: 6px 14px;
            border-radius: 16px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        .badge-prefill { background: rgba(139, 92, 246, 0.2); color: #a78bfa; }
        .badge-decode { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
        
        /* 公式说明 */
        .formula-box {
            margin-top: 20px;
            padding: 16px;
            background: rgba(99, 102, 241, 0.1);
            border-radius: 12px;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }
        .formula-title {
            color: #818cf8;
            font-size: 0.85rem;
            margin-bottom: 10px;
        }
        .formula {
            font-family: 'Times New Roman', serif;
            color: #e2e8f0;
            font-size: 1rem;
            text-align: center;
            padding: 10px;
        }
        .formula-desc {
            color: #94a3b8;
            font-size: 0.8rem;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Prefill vs Decode：两种计算模式的对比</h1>
        
        <div class="phase-container">
            <!-- Prefill 阶段 -->
            <div class="phase-card prefill-card">
                <div class="phase-header">
                    <div class="phase-icon">🚀</div>
                    <div class="phase-title">Prefill 预填充阶段</div>
                    <div class="phase-subtitle">一次处理完整 Prompt</div>
                </div>
                
                <div class="attention-viz">
                    <div class="viz-title">Attention 计算：完整的 N×N 矩阵（Causal Mask）</div>
                    <div class="matrix-container">
                        <div class="axis-label">Query<br>(N tokens)</div>
                        <div class="matrix-wrapper">
                            <div class="matrix prefill-matrix">
                                <!-- Row 1: t0 -->
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <!-- Row 2: t1 -->
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <!-- Row 3: t2 -->
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <!-- Row 4: t3 -->
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <!-- Row 5: t4 -->
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-mask">×</div>
                                <!-- Row 6: t5 -->
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                                <div class="matrix-cell cell-active">✓</div>
                            </div>
                            <div class="token-labels">
                                <div class="token-label">t₀</div>
                                <div class="token-label">t₁</div>
                                <div class="token-label">t₂</div>
                                <div class="token-label">t₃</div>
                                <div class="token-label">t₄</div>
                                <div class="token-label">t₅</div>
                            </div>
                            <div style="color:#64748b; font-size:0.65rem; margin-top:4px;">Key (N tokens)</div>
                        </div>
                    </div>
                </div>
                
                <div class="compute-indicator">
                    <div class="compute-row">
                        <span class="compute-label">计算量</span>
                        <div class="compute-bar">
                            <div class="compute-fill fill-compute" style="width:90%;">O(N²) 矩阵运算</div>
                        </div>
                    </div>
                    <div class="compute-row">
                        <span class="compute-label">访存量</span>
                        <div class="compute-bar">
                            <div class="compute-fill fill-memory" style="width:30%;">中等</div>
                        </div>
                    </div>
                </div>
                
                <div class="feature-list">
                    <div class="feature-item">
                        <span class="feature-icon">▸</span>
                        <span><strong>计算密集型</strong>：大量 GEMM 操作</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">▸</span>
                        <span>处理 N 个 prompt tokens</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">▸</span>
                        <span>生成并存储完整 KV Cache</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">▸</span>
                        <span>每个请求只执行一次</span>
                    </div>
                </div>
            </div>
            
            <!-- Decode 阶段 -->
            <div class="phase-card decode-card">
                <div class="phase-header">
                    <div class="phase-icon">🔁</div>
                    <div class="phase-title">Decode 解码阶段</div>
                    <div class="phase-subtitle">每次只处理 1 个新 Token</div>
                </div>
                
                <div class="attention-viz">
                    <div class="viz-title">Attention 计算：仅计算新 Token 的一行（1×N）</div>
                    <div class="matrix-container">
                        <div class="axis-label">Query<br>(1 token)</div>
                        <div class="matrix-wrapper">
                            <div class="matrix decode-matrix">
                                <!-- Row 1-6: 已缓存，不需重新计算 -->
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <div class="matrix-cell cell-cached">-</div>
                                <!-- Row 7: 新 token，只计算这一行 -->
                                <div class="matrix-cell cell-compute">✓</div>
                                <div class="matrix-cell cell-compute">✓</div>
                                <div class="matrix-cell cell-compute">✓</div>
                                <div class="matrix-cell cell-compute">✓</div>
                                <div class="matrix-cell cell-compute">✓</div>
                                <div class="matrix-cell cell-compute">✓</div>
                                <div class="matrix-cell cell-new">NEW</div>
                            </div>
                            <div class="token-labels">
                                <div class="token-label">t₀</div>
                                <div class="token-label">t₁</div>
                                <div class="token-label">t₂</div>
                                <div class="token-label">t₃</div>
                                <div class="token-label">t₄</div>
                                <div class="token-label">t₅</div>
                                <div class="token-label token-label-new">t₆</div>
                            </div>
                            <div style="color:#64748b; font-size:0.65rem; margin-top:4px;">Key (N+1 tokens)</div>
                        </div>
                    </div>
                </div>
                
                <div class="compute-indicator">
                    <div class="compute-row">
                        <span class="compute-label">计算量</span>
                        <div class="compute-bar">
                            <div class="compute-fill fill-compute" style="width:20%;">O(N)</div>
                        </div>
                    </div>
                    <div class="compute-row">
                        <span class="compute-label">访存量</span>
                        <div class="compute-bar">
                            <div class="compute-fill fill-memory" style="width:85%;">读取完整 KV Cache</div>
                        </div>
                    </div>
                </div>
                
                <div class="feature-list">
                    <div class="feature-item">
                        <span class="feature-icon">▸</span>
                        <span><strong>访存密集型</strong>：主要是 KV Cache 读取</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">▸</span>
                        <span>每次只处理 1 个新 token</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">▸</span>
                        <span>需要 PagedAttention 读取缓存的 KV</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">▸</span>
                        <span>循环执行，直到遇到终止条件</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 序列生成时间线 -->
        <div class="sequence-timeline">
            <div class="timeline-title">📊 序列生成过程的注意力计算演变</div>
            <div class="timeline-viz">
                <div class="seq-row">
                    <div class="seq-label">Prefill:</div>
                    <div class="seq-tokens">
                        <div class="seq-token tok-prefill">请</div>
                        <div class="seq-token tok-prefill">解释</div>
                        <div class="seq-token tok-prefill">什么</div>
                        <div class="seq-token tok-prefill">是</div>
                        <div class="seq-token tok-prefill">AI</div>
                        <div class="seq-token tok-current" title="采样第一个输出">→ ?</div>
                    </div>
                </div>
                <div class="seq-row">
                    <div class="seq-label">Decode 1:</div>
                    <div class="seq-tokens">
                        <div class="seq-token tok-prompt">请</div>
                        <div class="seq-token tok-prompt">解释</div>
                        <div class="seq-token tok-prompt">什么</div>
                        <div class="seq-token tok-prompt">是</div>
                        <div class="seq-token tok-prompt">AI</div>
                        <div class="seq-token tok-decode">人工</div>
                        <div class="seq-token tok-current">→ ?</div>
                    </div>
                </div>
                <div class="seq-row">
                    <div class="seq-label">Decode 2:</div>
                    <div class="seq-tokens">
                        <div class="seq-token tok-prompt">请</div>
                        <div class="seq-token tok-prompt">解释</div>
                        <div class="seq-token tok-prompt">什么</div>
                        <div class="seq-token tok-prompt">是</div>
                        <div class="seq-token tok-prompt">AI</div>
                        <div class="seq-token tok-prompt">人工</div>
                        <div class="seq-token tok-decode">智能</div>
                        <div class="seq-token tok-current">→ ?</div>
                    </div>
                </div>
                <div class="seq-row">
                    <div class="seq-label">Decode N:</div>
                    <div class="seq-tokens">
                        <div class="seq-token tok-prompt">请</div>
                        <div class="seq-token tok-prompt">...</div>
                        <div class="seq-token tok-prompt">...</div>
                        <div class="seq-token tok-prompt">...</div>
                        <div class="seq-token tok-prompt">...</div>
                        <div class="seq-token tok-prompt">...</div>
                        <div class="seq-token tok-decode">。</div>
                        <div class="seq-token tok-current">EOS</div>
                    </div>
                </div>
                <div class="phase-indicator">
                    <span class="phase-badge badge-prefill">紫色 = Prefill 阶段计算</span>
                    <span class="phase-badge badge-decode">橙色 = Decode 阶段计算</span>
                </div>
            </div>
        </div>
        
        <!-- 公式说明 -->
        <div class="formula-box">
            <div class="formula-title">💡 为什么 Decode 阶段是访存密集型？</div>
            <div class="formula">
                Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d) · V
            </div>
            <div class="formula-desc">
                在 Decode 阶段，Q 只有 1 行（当前 token），但 K 和 V 包含所有历史 token。计算量是 O(N)，但需要从显存中读取完整的 KV Cache（可能数 GB）。由于 GPU 的显存带宽远低于计算能力，瓶颈在于内存读取而非计算——这就是为什么 PagedAttention 能在 Decode 阶段发挥如此重要的作用。
            </div>
        </div>
    </div>
</body>
</html>
```

**为什么 Prefill 是计算密集型的？**

在 Prefill 阶段，模型需要处理完整的 Prompt 序列。假设 prompt 有 N 个 token，那么注意力计算涉及一个 N×N 的矩阵（经过 causal mask 后是下三角矩阵）。这意味着：Query 矩阵形状为 [N, d]，Key 矩阵形状为 [N, d]，注意力分数计算 QK^T 的复杂度是 O(N²d)。这是大量的矩阵乘法运算，GPU 的 Tensor Core 可以充分发挥作用，算力利用率很高。

**为什么 Decode 是访存密集型的？**

到了 Decode 阶段，情况完全不同。每次迭代只有 1 个新 token 需要处理，Query 矩阵变成了 [1, d]。虽然这个 token 仍然需要与之前所有 token 计算注意力（K 和 V 的形状是 [N, d]），但计算量从 O(N²d) 下降到了 O(Nd)。

然而，我们仍然需要从显存中读取完整的 KV Cache。对于一个有 1000 个 token 的序列，这可能意味着数百 MB 甚至数 GB 的数据读取。由于 GPU 的计算能力远超显存带宽（比如 A100 的算力是 312 TFLOPS，但显存带宽只有 2 TB/s），瓶颈转移到了内存访问上。

**这对优化意味着什么？**

理解这种差异对于优化推理性能至关重要。Prefill 阶段可以通过增大 batch size 来提高吞吐量，因为它是计算密集型的。而 Decode 阶段由于每次只处理一个 token，单个请求很难打满 GPU——这正是连续批处理如此重要的原因：它允许我们在 Decode 阶段把多个请求的 token "打包"在一起执行，充分利用 GPU 的并行能力。

在 nano-vllm 的 `model.py` 中，你可以观察到对这两种模式的不同处理方式。Prefill 时使用标准的 Flash Attention，而 Decode 时使用 PagedAttention 从非连续的物理块中读取 KV Cache。

---

### 五、动手实践：nano-vllm 源码入口指引

理论讲完了，现在让我们实际看看代码。nano-vllm 的代码结构非常清晰，以下是几个关键入口点：

**`engine.py` - 找到系统的心跳**

打开 `engine.py`，找到 `step()` 函数。这是整个系统的主循环驱动。你会看到它大概做这些事情：调用 scheduler 决定本轮执行哪些请求，准备模型输入（构建 input_ids、positions 等），调用 worker 执行前向传播，采样得到下一个 token，更新各请求的状态。

**`scheduler.py` - 理解调度决策**

查看 `_schedule()` 方法的实现。你会看到它如何遍历 waiting 队列，检查 BlockManager 是否有足够的空闲块，然后决定哪些请求可以进入执行。

**`block_manager.py` - 初探分页内存管理**

这里实现了 PagedAttention 的核心数据结构。重点关注 `block_table` 的结构和作用。每个请求都有自己的 block_table，记录着逻辑块到物理块的映射。

