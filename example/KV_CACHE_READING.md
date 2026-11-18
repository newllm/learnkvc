# `kv_cache.py` 阅读导图

这份代码实现的是 decoder-only Transformer 推理时的 **KV Cache**。它不是缓存 token，也不是缓存 attention 输出；它缓存每层注意力模块算出的 Key（K）和 Value（V）张量。下一次生成 token 时，只需要计算新 token 的 Q、K、V，再把新 K/V 追加到历史缓存中，避免重算整个前缀。

## 先建立整体图像

一次自回归生成的工作流如下：

```text
已有 prompt 的 K/V ───────┐
新 token ──> 投影出 Q,K,V ├─> K/V 拼接 ─> causal attention ─> 当前层输出
                            └─> 写回 KV cache
```

没有 Cache 时，生成第 `t` 个 token，需要重新对前 `t` 个 token 做每层 K/V 投影；有 Cache 时，只计算新 token 的 K/V，历史 K/V 直接复用。

## 张量形状：最重要的约定

文件采用的 K/V 布局是：

```python
(batch, kv_heads, sequence_length, head_dim)
```

例如：

```python
k.shape == (1, 2, 3, 8)
```

含义为：1 个请求，2 个 KV 头，当前缓存 3 个位置，每个头维度为 8。

`query` 的形状类似，不过第二维是 `q_heads`。当 `q_heads > kv_heads` 时，这是 GQA/MQA；本实现用 `repeat_interleave` 将 K/V 按头复制，以便与 Q 做注意力计算。

## 建议的阅读顺序

### 1. `LayerKVCache`：一层的历史状态

从 `LayerKVCache` 开始。一个 Transformer layer 有自己独立的一组 K/V；因此类中只有：

```python
self.key
self.value
```

初始时它们都是 `None`。`length` 读取倒数第二维，即缓存的 sequence length。

### 2. `append()`：Cache 的核心操作

`append(new_key, new_value)` 逻辑可以概括为：

```text
空缓存：              cache = new_kv
已有历史：            cache = cat(old_kv, new_kv, seq 维)
开启窗口：            cache = cache 的最后 max_length 个位置
```

这里的拼接维度是 `dim=-2`，也就是 `sequence_length` 维度。不要沿 `head_dim` 或 head 数量的维度拼接。

调用后，方法既更新内部 `self.key/self.value`，也返回完整缓存，供当前注意力计算直接使用。

### 3. `KVCache`：把多层缓存装在一起

真实 Transformer 有多层。因此 `KVCache(num_layers)` 创建：

```python
[LayerKVCache(...), LayerKVCache(...), ...]
```

模型执行到第 `layer` 层时调用：

```python
all_key, all_value = cache.update(layer, new_key, new_value)
```

这就是把某一层刚投影出的 K/V 写进它自己的缓存。

### 4. `cached_attention()`：Cache 如何接到注意力计算

这是最值得逐行看的函数。它分为四步：

1. 在更新前读取 `past`，即当前层已有的历史长度。
2. 通过 `cache.update(...)` 拼接新 K/V。
3. 若使用 GQA/MQA，把 K/V 扩展到与 Q 相同的 head 数。
4. 用 `Q @ K^T / sqrt(head_dim)` 得分，施加 causal mask，softmax 后乘 V。

数学上对应：

```text
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

其中的 `K` 和 `V` 已经包含历史部分和本次新加入的部分。

## 为什么不能只写 `is_causal=True`

在纯 prefill 时，Q 和 K 都从位置 0 开始，普通下三角 mask 就够用。

但 Cache 下可能是：已有 100 个历史位置，本次同时处理 3 个新 token。此时本次 Q 的绝对位置是 100、101、102，而 K 的位置是 0 到 102。第一条新 Q 可以看 0 到 100，不能看 101、102。

代码使用：

```python
past = cache.layers[layer].length
```

再根据 Q 和 K 的相对位置手工建立 mask，保证“未来 token 不可见”。这也是 `cached_attention()` 不直接调用高阶 attention API 的原因：显式写出来更容易看清带历史 Cache 时的位置对齐。

## 滑动窗口为什么要重新计算 key 的起点

`max_length=128` 时，只保留最后 128 个 K/V。例如已有 128 个位置后再加入 3 个：最旧的 3 个会被淘汰。此时缓存张量的第 0 项不再对应绝对位置 0。

所以代码计算：

```python
first_key_position = past + q_len - k_len
```

这将“缓存内的相对下标”映射回本轮计算的绝对位置，避免多 token 追加时意外让较早的 Q 看到未来的新 K。

## `reorder()` 与 `fork()` 分别解决什么问题

`reorder(batch_indices)` 用于 beam search。beam 每一步会筛选、交换或复制候选序列，所以所有层的 cache 也必须以相同方式重排。

`fork()` 用于从同一前缀向不同后缀分叉。它调用 `clone()`，所以之后对其中一条分支追加 K/V，不会改到另一条分支。

```text
共享前缀 cache
   ├── fork ──> 分支 A 继续生成
   └── fork ──> 分支 B 继续生成
```

## 用入口示例验证理解

文件末尾的 `__main__` 分两次调用：

1. 输入 3 个 token 的 Q/K/V，得到长度为 3 的缓存。
2. 再输入 1 个 token 的 Q/K/V，缓存长度变成 4。

第二步没有重新提供前 3 个 token 的 K/V；这正是 Cache 节省计算的地方。

## 阅读时可检查的五个问题

1. 为什么 K/V 沿 `dim=-2` 拼接？
2. 为什么每层要有独立的 `LayerKVCache`？
3. `past` 为什么必须在 `cache.update()` 前读取？
4. GQA 中为什么 Q 的头数必须是 KV 头数的整数倍？
5. 开启窗口后，为什么不能把缓存下标直接当成绝对 token 位置？

如果这五点都能用代码回答，就已经掌握了这份实现的关键路径。

## 这份示例刻意没有处理的生产问题

- 预分配固定大小的 GPU buffer，避免每次 `torch.cat` 分配和拷贝。
- 不同请求长度的 padding、attention mask 和 continuous batching。
- RoPE 的 position ids，以及滑动窗口后的位置策略。
- 显存分页、KV 量化、跨请求共享前缀。
- 使用 FlashAttention / `scaled_dot_product_attention` 等高性能 kernel。

因此它适合理解正确性和数据流；生产推理引擎通常会用 paged KV cache 或预分配缓存来提高吞吐量。
