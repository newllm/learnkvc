# learnkvc 交互式阅读器 · 设计稿

把「阅读导图」和「示例代码」合并成同一个页面：**左边代码、右边讲解、底部时间轴**，每一步都能看到「代码执行到哪一行、内部状态变成什么样」，还能改参数、看计数器、回放动画。

**v2（本轮）把主线从「读」改成「想」**：先让用户预测、归因，再把代码与 trace 作为证据揭晓，最后要求用自己的话总结。知识不是目标，思考、反思、总结才是。

关键约束：讲解文字和代码都从仓库现有文件生成，不在 HTML 里手抄；页面零依赖、可离线、双击 `index.html` 就能开。

## 1. 目标与非目标

**目标**

- 读代码 → 看状态 → 改参数，形成闭环；每个结论都能在页面上「跑」一遍。
- 一份事实来源：`example/*.py` 是代码的唯一来源，`example/*_READING.md` 是讲解的唯一来源，`README.md` 是总览的来源。
- 可深链：`#kv/step-3`、`#kv/L186` 这种 URL 能分享到具体步骤/代码行。
- 键盘可用：`←/→` 步进、`空格` 播放/暂停、`Home/End` 跳首尾。

**非目标**

- 不做通用 IDE / 在线 playground，不做任意代码执行。
- 不在浏览器里跑 Python（trace 由 Python 预生成，见 §6.3）。
- 不引入前端框架、npm、CDN。

## 2. 设计原则

1. **问题驱动**：每个知识块之前先放一个「先想一想」；用户没表态，不展开解释。
2. **以证据代替结论**：代码、trace、计数器是证据；正文只回答用户已经产生的疑问。
3. **单一事实来源**：代码、讲解、数据分别来自 `example/*.py`、`*_READING.md`、构建脚本；HTML 只做渲染。
4. **渐进增强**：没有 JS 也能读完整讲解（服务端渲染好的 `<noscript>` 版本或静态章节页）。
5. **零运行时依赖**：生成物 `data.js` 提交进仓库，打开即用，`file://` 也能工作（用 `<script>` 注入数据，不用 `fetch`）。
6. **可链接**：应用状态（章节/步骤/高亮行/参数）全部写进 URL hash。
7. **性能预算**：首屏 < 300 KB，全量 < 1.5 MB，无网络请求。

## 3. 信息架构

当前入口顺序是「先想 → 三章四段式（含多次提问、三层归因） → 模型解剖 / 计算器 → 后测 → 反馈 → 作业」；多次提问与归因不再单独成页：

| 视图 | 作用 | 内容来源 |
| --- | --- | --- |
| 先想 | 6 个预测题，覆盖一次提问 / 多轮 / 多请求 / 三层分工 | `web/tools/pedagogy.json` |
| ① 探索（第 1 章） | 14 步代码 ↔ 讲解联动；每步先放「先想一想」再展开证据 | `kv_cache.py` + `KV_CACHE_READING.md` |
| ② Prefix 章的「多次提问」步骤 | 多轮对话 / 多请求共享前缀模拟器，现场比较 prefill 与 KV 占用 | `traces.reuse` |
| 每章「总结归纳」内嵌归因 | 3 张卡分到「模型 / GPU / 引擎」，提交后才能完成本章 | `pedagogy.json` |
| ④ 模型解剖 | 模型组成、参数/维度/层数、参数如何参与 token 生成、GPU 加载、kernel 表 | `anatomy.json` + 公式 |
| P2 / P3（规划） | Prefix cache、PagedAttention 沿用同一套「先想 → 证据」结构 | 后续 |

总览不再是独立页面：`README.md` 的「为什么需要 KV Cache」收进「先想」页的折叠区，作为忍不住想看结论时的证据。

## 3.5 教学逻辑 v2：预测 → 证据 → 归因 → 表达

这条主线回答三个问题，也是页面上所有交互存在的理由：

1. **一次提问时，KV cache 省了什么？** → 先猜 decode 每步做什么，再用 92 vs 15 的计数器和双栏播放验证。
2. **多次提问时，它又多出什么作用？** → 多轮对话 vs 多请求共享前缀，亲手加一轮/加一个请求，观察「省计算」和「省显存」不是一回事。
3. **模型、GPU、引擎各负责哪一段？** → 把 12 张卡片分到三层，再用自己的话解释一次。

四个环节：

| 环节 | 用户要做的动作 | 页面给的反馈 | 防止的事 |
| --- | --- | --- | --- |
| 先想 | 从选项里挑一个（承诺答案） | 立刻标出对/错 + 一小段「参考」 | 不思考直接翻答案 |
| 探索 | 点「先想一想」展开，再读代码/证据 | 代码高亮、trace、计数器 | 把正文当课文背 |
| 多次提问（Prefix 章内） | 加一轮对话 / 加一个请求 | prefill 与 KV 的对比条 | 把多轮和多请求混为一谈 |
| 归因（每章总结内） | 3 张卡分类、提交 | 逐条理由 + 参考总结（先写后看） | 只会复述、不会归因 |

三条硬规则：

- **先表态，再揭晓**：预测题和「先想一想」都不默认展开；展开动作由用户触发。
- **不评判人，只对照**：选错只标红并给参考理由，不算分；归因练习才给命中数。
- **留下自己的话**：总结写进 `localStorage`，可复制；参考总结只有点了「对照参考」才出现。

这一层和 P0/P1 的关系：**代码联动、计数器、缓存网格仍然是必要的证据层**，但它们从「主角」降级为「验证用户假设的工具」。

## 3.55 学习闭环：设定 → 前测 → 学习 → 后测 → 反馈（v2.2）

用户需要先知道「这次学的是哪个模型、跑在什么 GPU 上」，否则后面的显存/带宽数字没有着力点。

- **本次学习设定**（先想页顶部）：锚点模型 Qwen3-Coder-30B-A3B 的规格卡（30.53B / 3.35B 激活、48 层、hidden 2048、32/4 头、128 expert、256K 上下文、96 KiB/token、Q4 18.5 GB）；GPU 可选 RTX 4090/3090/5090、A100 80GB、Apple M2 16GB，展示显存/带宽/BF16 算力/PCIe，并实时给出「Q4 + 32K KV + 运行时 ≈ 23 GB → 能否跑、decode 理论上限、prefill 计算下限」。
- **前测（评分）**：6 题，选完点提交才显示对错和解释；分数存 `localStorage`，作为基线。避免「边做边看答案」。
- **后测（评分）**：同概念换问法的 6 题，提交后生成反馈。
- **问题两层**：每题都有「考什么」一句话概要 + 可展开的「背景 / 怎么想」详细描述，帮助理解题意但不泄露答案。
- **反馈页**：前测/后测分数、变化、按概念对照（提升/维持/退步/仍需复习）、学习完成度（探索步骤数、访问视图数）、错题回看按钮（直达对应章节步骤/视图）；最后的总结也是「概要 + 详细描述」两层。
- 完成度记录：`app.js` 在切换视图/步骤时写 `LK.progress`（`learnkvc.progress.v1`）。

## 3.58 四段式章节 + 迁移作业（v2.4）

**v2.5 收束**：取消独立的「多次提问」「归因与总结」页——多次提问作为 Prefix 章第 11 步内嵌模拟器；三层归因拆成每章 3 张卡放进「总结归纳」，提交归因后才能完成本章；跨章归因综合放在课后作业的三章对照矩阵与迁移题里。

每章不再只是「一组阅读步骤」，而是一个完整学习单元：

| 阶段 | 内容 | 完成条件 |
| --- | --- | --- |
| 先想 | 3 道预测题（选完才揭示，不计分） | 3 题都想过 → `lessons.<ch>.think` |
| 探索阅读 | 现有 steps：代码 ↔ 讲解 ↔ 可视化 | 记录 `exploreMax` |
| 提问反思 | 2–4 个开放问题 + 可折叠参考；写入本地 | 点「完成反思」→ `reflect` |
| 总结归纳 | 参考总结（概要+详细）+ 用户自己的总结 | 点「完成本章」→ `summary` |

- 视图顶部 stagebar 显示四阶段与完成状态；非探索阶段隐藏代码 pane / 可视化 / 时间轴。
- 三章 `summary` 全为 true 后解锁 `#homework`：三章对照矩阵（复用单位 / 共享 / 淘汰 / 物理映射 / 责任层）+ 4 道迁移题（CPU cache、DB buffer pool、CDN、Git packfile、浏览器缓存任选其一；KV 公式；TTL 放哪层；自由类比）+ 参考答案 + 提交记录（`localStorage`）。
- 未完成时显示锁定页：每章状态 + 「去学习」按钮。

## 3.6 模型解剖（v2.1）

用户需要先有「模型长什么样」的直觉，否则 KV cache 只是名词。新增第 4 个视图，仍按「先猜 → 证据」组织：

| 小节 | 先问什么 | 给什么证据 |
| --- | --- | --- |
| 组成 | 一个 LM 由哪几块组成？ | embedding → N 层 → norm → lm_head 的 stack，配参数拆分表 |
| 层与维度 | 48 层的第 1 层和第 48 层输出一样宽吗？ | 层网格 + 单层形状 + 可调「层数 / KV 头数」的参数量/KV 计算器 |
| 参数 → token | 参数在生成一步里怎么被用到？ | 8 步流水线：embed → QKV → RoPE → KV 写入 → attention → o_proj → MoE → head，标形状/kernel/谁负责 |
| GPU 加载 | 权重从磁盘到显存经过哪几站？ | 磁盘 → RAM → H2D → 显存四张卡，带体积和 25 GB/s 估算 |
| kernel 表 | 生成本步调用了哪些 GPU 函数？ | 11 行表：阶段 / 操作 / cuBLAS / FlashAttention / reduction，prefill 与 decode 分开 |
| 自问 | 层数、参数、维度、KV 的关系 | 三张折叠卡，回到「模型 / GPU / 引擎」 |

预设覆盖 TinyModel、Qwen2.5-Coder-7B/32B、Qwen3-Coder-30B-A3B；参数量用公式在构建期算好并断言（30B → 30.53B total / 3.35B active / 96 KiB per token）。

## 4. 页面布局

```
┌──────────────────────────────────────────────────────────────────────┐
│ learnkvc · 交互式阅读     总览  KV  Prefix  Paged  计算器   [暗/亮] │
├──────────────────────────┬───────────────────────────────────────────┤
│ 代码（sticky，可滚动）   │ 讲解卡（跟随当前 step）                    │
│                          │ ┌───────────────────────────────────────┐ │
│  1  @dataclass           │ │ step 3 / 16 · 有 cache 的 prefill      │ │
│  2  class LayerKVCache:  │ │ 正文来自 KV_CACHE_READING.md 对应小节  │ │
│  3      ...              │ │ [跳转到源码] [上一篇] [下一篇]         │ │
│                          │ └───────────────────────────────────────┘ │
│  高亮当前 step 的行      │ ┌ 可视化区（按章节不同） ──────────────┐ │
│  点行号复制行链接        │ │  token 条 / 缓存网格 / block table    │ │
│                          │ │  / 块池 / 计算器滑杆                  │ │
│                          │ └───────────────────────────────────────┘ │
├──────────────────────────┴───────────────────────────────────────────┤
│ ◀ ▮▶ 进度条  1·2·3…16     计数器：前向位置 15 · attention 对数 240   │
└──────────────────────────────────────────────────────────────────────┘
```

移动端（< 960px）：代码区折叠为「查看当前步骤代码」抽屉，可视化在上、讲解在下，时间轴固定在底部。

## 5. 交互设计（逐章）

### 5.1 总览：为什么需要 cache

- 左右两栏 token 流：左「无 cache」每步重喂整段前缀，右「有 cache」prefill 后每步只喂 1 个 token。
- 播放时同步增长输入长度（8 → 9 → … vs 8 → 1 → 1），并累加两个计数器：`位置前向次数`、`attention 对数`。
- 点击任一 token，高亮它在右侧缓存中的位置；对应的 `README.md` 段落以 tooltip 形式出现。

### 5.2 第 1 章：单请求 KV cache

- **步进器**：step 来自 `kv_cache.py` 的 `generate()`/`Step` 数据（prompt 8 个 token、生成 8 步）。
- **双栏代码**：同一 step 同时高亮「无 cache 分支」和「cache 分支」两段代码（`full_attention` vs `cached_attention`），差异用底色区分。
- **缓存网格**：2 层 × 序列位置；每格显示 K/V 是否已写入；GQA 展开（4 个 KV 头 → 8 个 Q 头）用连线表示。
- **特性开关**：滑动窗口（`window`）、`reorder()`（beam search）、`fork()`（分支）三个按钮，复用 `check_cache_features()` 的断言结果，展示「缓存搬运/复制」的代价。
- **计数器**：`位置前向`、`attention 对数`、`白白重算的位置`；鼠标悬停显示公式，点击可展开与 `print_summary` 一致的对照表。

### 5.3 第 2 章：多请求 Prefix cache

- **场景选择**：共享前缀 / 池子小触发淘汰 / 相同短语不同位置（对应 `scenario_shared_prefix`、`scenario_eviction`、`scenario_repeated_phrase`）。
- **请求时间线**：请求 A → B → C 的 prompt 切成 `system prompt / few-shot / 问题` 三段，颜色区分「命中缓存」「新 prefill」。
- **块池**：`NUM_BLOCKS` 个槽位（默认 8），每格显示块号、hash 前 4 位、引用计数、状态（空闲/共享/待淘汰）。
- **hash 链**：点击块显示 `hash(parent, tokens)` 的父块指针，连成前缀树；与 `PREFIX_CACHE_READING.md` 的「身份」一节联动。
- **播放 acquire / append / release**：块在「新分配 → 命中 → 共享 → 叶子淘汰」之间变色；LRU 淘汰时高亮被淘汰的叶子。
- **指标**：prefill 省下的 token 数、命中率、淘汰次数；与 `print_summary` 的输出逐项核对。

### 5.4 第 3 章：PagedAttention

- **三列布局**：logical token（0..N）→ block table（逻辑块 → 物理块）→ 物理块池（`BLOCK_SIZE=4`）。
- **append 动画**：token 逐个落格，块从空闲变占用；跨块时高亮 block table 新增一行。
- **gather 动画**：点击某个 query 位置，按 block table 高亮它读取的所有物理块，演示「逻辑连续、物理可以不连续」。
- **对比模式**：连续布局（预留 `max_len`，浪费条）vs 分页（按需分配）；实时计算碎片率 `1 - used / (num_blocks × block_size)`。
- **序列时间线**：两个序列交错、释放其中一个、第三个序列复用它的块（对应 `main()` 的流程），每一步断言 paged 输出 == 连续布局输出。

### 5.5（可选）显存/算力计算器

- 输入：模型 preset（Qwen3-Coder-30B-A3B / Qwen2.5-Coder-7B/14B/32B / Qwen3.5-4B/9B）、量化、上下文长度、KV dtype、并发数、引擎。
- 输出：`权重 + KV + 激活 + 运行时` 的堆叠条，以及 24/32/48 GB 卡的结论；prefill 的 FLOPs 拆成 `GEMM` 与 `full attention` 两条腿，附估算耗时。
- 公式与 `macos/ollama/README.md` 的表格一致，悬停显示推导；preset 参数硬编码在 `data/presets.js`，便于和文档同步。

### 5.6（可选）模型解剖

- 横向流程图：`RMSNorm → q/k/v + q_norm/k_norm → RoPE → KV 写入 → GQA → o_proj → 残差 → RMSNorm → router → top-8 experts → 合并 → 残差`。
- 每个节点可点开：形状（n=10K / n=1 切换）、FLOPs、以及对应的源码/文档出处。
- prefill/decode 切换时，张量形状从 `[n, d]` 变成 `[1, d]`，直观看出「大 GEMM → GEMV」。

## 6. 代码 ↔ 文档 ↔ 状态 同步（核心机制）

### 6.1 step 模型

每章由一组有序 step 组成，`web/tools/steps.json` 手写这份清单：

```json
{
  "id": "kv/decode-cached",
  "chapter": "kv",
  "title": "有 cache：每步只喂 1 个新 token",
  "codeRefs": ["function:generate", "function:LayerKVCache.append"],
  "docRefs": ["KV_CACHE_READING.md#第-2-部分缓存的数据结构"],
  "viz": "kv-generation",
  "state": { "tokens": 16, "counters": { "forward": 15, "pairs": 240 } }
}
```

- `codeRefs` 用**符号**而不是行号：`function:generate`、`class:LayerKVCache.append`，由构建脚本解析成行区间。
- `docRefs` 指向阅读导图的标题锚点，构建时按 `##`/`###` 切分后校验存在。
- `viz` 决定渲染哪个可视化组件；`state` 是 trace 里该 step 的状态快照。

### 6.2 行号解析：用 AST，不手写行号

`web/tools/build.py` 用 `ast` 解析每个示例文件：

- `function:generate` → `ast.FunctionDef.lineno .. end_lineno`
- `class:LayerKVCache.append` → 类内方法的行区间
- 需要更细的区间时，在 `steps.json` 里补充 `"lines": [364, 407]` 兜底；示例源码里**不加任何标记注释**，保持可独立运行、干净可读。

代码改动后重跑构建即可，`steps.json` 不会因为插了几行注释而失效。

### 6.3 trace 生成：Python 预生成，浏览器只回放

`web/tools/collect_traces.py` 复用示例本身的类和函数，逐步导出状态：

- `kv_cache.py`：直接复用 `generate()` 返回的 `Step`/`Stats`，无需改动示例。
- `prefix_cache_demo.py`：实例化 `Simulator`，在每个 `acquire/append/release` 后快照块池（块号、hash、引用计数、LRU 顺序）。
- `paged_attention.py`：按 `main()` 的序列重放 `feed()`，记录每步的 `block_table`、池占用、paged vs dense 的断言结果。

输出 `web/data/traces.js`（`window.__TRACES__ = {...}`）并提交仓库——查看者不需要 Python，也不需要联网。重新生成需要 PyTorch（第 1、3 章）；第 2 章纯标准库。

### 6.4 讲解解析：构建期把 Markdown 变成卡片

`build.py` 读取三份 `*_READING.md`：

- 按 `##` / `###` 切成 section，记录 heading → anchor；
- 用一个小型 Markdown 渲染器（标题/段落/列表/代码块/表格/引用/加粗/行内代码/链接）转成 HTML；
- step 的 `docRefs` 关联到 section；未关联的 section 收进「全文模式」（右侧可滚动全文，点击标题跳到代码）。

不手写讲解文字，正文与仓库文档永远一致。

## 7. 数据管线

```
example/*.py ──ast──► 代码文本 + 符号→行号 ─┐
example/*_READING.md ──mini-md──► 讲解 HTML ─┼──► web/data/data.js ──► index.html
web/tools/steps.json ──(step 清单)───────────┘
example/*.py ──collect_traces.py──► traces ──┘
```

- `data.js` 是生成物，提交进仓库；`index.html` 双击可用。
- 构建时做校验：每个 `codeRef` 能解析、每个 `docRef` 有对应标题、trace 的 step id 与 `steps.json` 一一对齐、第 1 章计数器与 `print_summary` 的断言值一致。
- 文档改动后重跑 `python3 web/tools/build.py`；CI 加 `--check` 可防漂移。

## 8. 技术栈与目录

```
web/
├── index.html
├── DESIGN.md
├── assets/
│   ├── style.css          # 暗色默认 + 亮色变量 + 响应式
│   ├── app.js             # hash 路由、章节装配、step 播放器
│   ├── codeview.js        # 自写 Python tokenizer、行号、行高亮、行链接
│   └── viz/
│       ├── overview.js
│       ├── kv.js
│       ├── prefix.js
│       ├── paged.js
│       ├── calculator.js
│       └── anatomy.js
├── data/
│   ├── data.js            # 生成：代码 + 文档 HTML + steps（提交）
│   ├── traces.js          # 生成：第 1/2/3 章状态快照（提交）
│   └── presets.js         # 手写：计算器的模型参数（与文档同步）
└── tools/
    ├── build.py
    ├── collect_traces.py
    └── steps.json
```

- **无框架**：原生 DOM + SVG + CSS；可视化组件挂到 `window.LK.viz.<name>`，由 app.js 按 `step.viz` 调用。
- **无模块加载器**：`<script>` 顺序加载（`data.js` → `traces.js` → `viz/*.js` → `app.js`），保证 `file://` 可用；需要 `fetch` 时再提供 `python3 -m http.server` 的说明。
- **语法高亮**：自写约 80 行 Python tokenizer（关键字/字符串/注释/数字/装饰器/函数名），避免引入 highlight.js；颜色走 CSS 变量，与暗/亮主题一致。
- 图表用 SVG/DOM，动画用 CSS transition + `requestAnimationFrame`。

## 9. 视觉与可用性

- **配色语义固定**：prompt token、生成 token、命中缓存、新分配块、被淘汰块各一套颜色；色盲安全模式下加纹理/形状区分。
- **字体**：代码等宽（系统 `ui-monospace`），正文系统无衬线；行高 1.5–1.6。
- **动效**：200–300ms；`prefers-reduced-motion: reduce` 时直接跳变。
- **无障碍**：`aria-live="polite"` 播报计数器；时间轴是原生 `<button>`；焦点环可见；对比度 ≥ 4.5:1。
- **主题**：暗色默认，可切亮色，选择存 `localStorage`。
- **打印**：`@media print` 隐藏控件，输出「讲解 + 代码 + 当前状态」。
- **深链**：`#kv/step-3`、`#kv/L186`、`#calculator?ctx=32k&kv=fp16`。

## 10. 实施计划

| 阶段 | 内容 | 产出 |
| --- | --- | --- |
| P0 | `build.py` + 页面壳 + 代码高亮 + 第 1 章文档联动（无动画） | 能双击打开、左右联动 |
| P1 | 第 1 章：双栏播放、计数器、缓存网格、滑动窗口/reorder/fork | 第 1 章完整 |
| P1.5 | 思考层：先想 6 题、每步先想一想、多次提问模拟器、归因 + 总结 | 主流程改为「先想 → 证据 → 归因」 |
| P1.6 | 模型解剖：组成/层与维度/参数→token/GPU 加载/kernel 表 + 4 个预设 | 补上「模型本身」的直觉 |
| P2 ✅ | 第 2 章：prefix 场景 + 块池 + LRU 动画 | 已完成 |
| P3 ✅ | 第 3 章：block table + append/gather 动画 + 碎片对比 | 已完成 |
| P4 ✅ | 计算器 + 模型解剖 + 亮色主题 + 响应式 + 无障碍 | 已完成（键盘/跳转链接/aria-live/主题记忆） |
| P5 ✅ | README 入口、GitHub Pages 部署（`.github/workflows/pages.yml`） | 已完成，待仓库开启 Pages |

每阶段独立可交付：P0 完成就能替代「纯 Markdown 阅读」，后续只是加交互。

## 11. 风险与取舍

| 风险 | 取舍 |
| --- | --- |
| JS 重实现示例逻辑会产生偏差 | 用 Python 预生成 trace，浏览器只回放；示例仍可独立运行 |
| 代码/行号漂移 | `steps.json` 用 AST 符号引用，不写死行号；CI `--check` |
| trace 生成依赖 PyTorch | 只在构建时依赖；生成物提交后，查看者零依赖 |
| `file://` 下 ES modules / fetch 被拦 | 全部用普通 `<script>` + 全局命名空间；数据内联生成 |
| 文档 Markdown 语法超出迷你渲染器 | 只支持仓库实际用到的语法，超出的降级为 `<pre>`，构建时报 warning |
| 体积膨胀 | 代码 + 文档 + trace 预计 300–600 KB；gzip 后更小 |
| 维护成本 | 改文档/代码后重跑一个命令；UI 组件与数据解耦，新增章节只加一个 `viz/*.js` |

## 12. 验收标准

1. 断网、双击 `web/index.html` 可完整使用。
2. 每个 step 的代码高亮、讲解卡、可视化状态三者一一对应。
3. 第 1 章计数器与 `python3 example/kv_cache.py` 的输出一致；第 2/3 章的断言在 trace 里保留并可展示。
4. 分享 `#kv/step-5` 打开后定位到同一状态。
5. 960px 以下布局可用，键盘可完成全部步进操作。
