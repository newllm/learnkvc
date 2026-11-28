# whykvc 交互式阅读器 · 设计稿

把「阅读导图」和「示例代码」合并成同一个页面：**左侧步骤导航（可收缩）、中间代码、右侧讲解，底部保留前后步与计数器**，每一步都能看到「代码执行到哪一行、内部状态变成什么样」，还能改参数、看计数器、回放动画。

**v2（本轮）把主线从「读」改成「想」**：先让用户预测、归因，再把代码与 trace 作为证据揭晓，最后要求用自己的话总结。知识不是目标，思考、反思、总结才是。

关键约束：讲解文字和代码都从仓库现有文件生成，不在 HTML 里手抄；页面零依赖、可离线、双击 `index.html` 就能开。

## 1. 目标与非目标

**目标**

- 读代码 → 看状态 → 改参数，形成闭环；每个结论都能在页面上「跑」一遍。
- 一份事实来源：`example/*.py` 是代码的唯一来源，`example/*_READING.md` 是讲解的唯一来源，`README.md` 是总览的来源。
- 可深链：`#kv/3`、`#kv/L186` 这种 URL 能分享到具体步骤/代码行。
- 键盘可用：`←/→` 步进、`空格` 播放/暂停、`Home/End` 跳首尾。

**非目标**

- 不做通用 IDE / 在线 playground，不做任意代码执行。
- 不在浏览器里跑 Python（trace 由 Python 预生成，见 §6.3）。
- 不引入前端框架、npm、CDN。

## 2. 设计原则

1. **问题驱动**：每个知识块之前先放一个「请回答」；用户没表态，不展开解释。
2. **以证据代替结论**：代码、trace、计数器是证据；正文只回答用户已经产生的疑问。
3. **单一事实来源**：代码、讲解、数据分别来自 `example/*.py`、`*_READING.md`、构建脚本；HTML 只做渲染。
4. **渐进增强**：没有 JS 也能读完整讲解（服务端渲染好的 `<noscript>` 版本或静态章节页）。
5. **零运行时依赖**：生成物 `data.js` 提交进仓库，打开即用，`file://` 也能工作（用 `<script>` 注入数据，不用 `fetch`）。
6. **可链接**：应用状态（章节/步骤/高亮行/参数）全部写进 URL hash。
7. **性能预算**：首屏 < 300 KB，全量 < 1.5 MB，无网络请求。

## 3. 信息架构

当前入口顺序是「请回答 → 认识 GPU（GPU / NPU 与国产算力 / CUDA 执行模型，三章三段式） → 认识 LLM（Transformer 与 LLM / LLM 结构 / LLM 训练） → 显存计算器（顶栏一级入口） → 认识 KVCache（三章三段式） → 推理网络（总览 / NCCL 学习章节 / ECN + DCQCN，三章三段式） → LLM 优化（三段式） → 后测 → 反馈」；多次提问与归因不再单独成页：

| 视图 | 作用 | 内容来源 |
| --- | --- | --- |
| 请回答 | 10 题前测，覆盖 LLM 构成 / 一次提问 / 多轮 / 多请求 / 三层分工等 | `web/tools/pedagogy.json` |
| ① 探索（第 1 章） | 14 步代码 ↔ 讲解联动；每步先放「请回答」再展开证据 | `kv_cache.py` + `KV_CACHE_READING.md` |
| ② Prefix 章的「多次提问」步骤 | 多轮对话 / 多请求共享前缀模拟器，现场比较 prefill 与 KV 占用 | `traces.reuse` |
| 每章「反思与总结」内嵌归因 | 3 张卡分到「LLM / GPU / 引擎」；归因 + 至少 2 条反思 + 我的总结 + 章节测验提交后才能完成本章 | `pedagogy.json` |
| ④ LLM 结构 | 三段式：LLM 组成、参数/维度/层数、参数如何参与 token 生成、GPU 加载、kernel 表、GPU 函数地图、运行时、结构仿真 | `anatomy.json` + `anatomy.js` + `structure_sim.py` |
| ④b LLM 训练 | 三段式章节：训练 step、显存账本、并行/省显存、最小训练脚本 + 每卡显存计算器 | `train.json` + `train_tiny.py` |
| ④c 推理网络 · 总览 | 三段式章节：NCCL collectives/ring 模拟、RoCE/RDMA、PD 分离 + KV 传输计算器 | `net.json` + `net.js` |
| ④d NCCL 学习章节 | 三段式章节：collectives、ring/tree 逐步播放、α-β 扫描与 crossover、拓扑、排查清单 + 仿真代码 | `nccl.json` + `nccl.js` + `nccl_sim.py` |
| ④e ECN + DCQCN | 三段式章节：ECN/WRED 标记与 CNP、DCQCN 速率收敛、交换机组网/配置、服务器侧 RoCEv2/PFC/ECN、无损策略、RDMA 业务优先级/PFC、ECN 优先级/阈值、排障 | `ecn.json` + `ecn.js` + `ecn_sim.py` |
| ④f 认识 GPU · GPU | 三段式章节：CPU/GPU/NPU 分工、CPU vs GPU 延迟/吞吐、LLM 适配、GPU 取代不了 CPU、显存层级与 L1/L2 cache、访存模式、CUDA/ROCm/昇腾/HCU 生态 | `gpu.json` + `gpu.js` + `gpu_sim.py` |
| ④f2 认识 GPU · NPU 与国产算力 | 三段式章节：GPU/NPU/ASIC 执行模型、国际/国产厂商地图、软件栈、LLM 选型 | `npu.json` + `npu.js` + `npu_sim.py` |
| ④g 认识 GPU · CUDA 执行模型 | 三段式章节：grid/block/warp/SM、occupancy 计算器、内存层级、coalescing、roofline、CUDA Graph + 字节账 | `cuda.json` + `cuda.js` + `cuda_sim.py` |
| ④h Transformer 与 LLM | 三段式章节：自注意力、架构、时间线、缩放定律、为什么 LLM 胜出、边界 + 账本仿真 | `transformer.json` + `transformer.js` + `transformer_sim.py` |
| ④i LLM 优化（下拉：总览 / FlashAttention / 量化 / 投机解码锚点） | 三段式章节：瓶颈地图、FlashAttention、PagedAttention、量化、投机解码、调度/结构优化、选型决策 + 计算器 | `opt.json` + `opt.js` + `opt_sim.py` |
| P2 / P3（规划） | Prefix cache、PagedAttention 沿用同一套「请回答 → 证据」结构 | 后续 |

总览不再是独立页面：`README.md` 的「为什么需要 KV Cache」收进「请回答」页的折叠区，作为忍不住想看结论时的证据。

## 3.5 教学逻辑 v2：预测 → 证据 → 归因 → 表达

这条主线回答三个问题，也是页面上所有交互存在的理由：

1. **一次提问时，KV cache 省了什么？** → 先猜 decode 每步做什么，再用 92 vs 15 的计数器和双栏播放验证。
2. **多次提问时，它又多出什么作用？** → 多轮对话 vs 多请求共享前缀，亲手加一轮/加一个请求，观察「省计算」和「省显存」不是一回事。
3. **LLM、GPU、引擎各负责哪一段？** → 把 12 张卡片分到三层，再用自己的话解释一次。

四个环节：

| 环节 | 用户要做的动作 | 页面给的反馈 | 防止的事 |
| --- | --- | --- | --- |
| 请回答 | 从选项里挑一个（承诺答案） | 立刻标出对/错 + 一小段「参考」 | 不思考直接翻答案 |
| 探索 | 点「请回答」展开，再读代码/证据 | 代码高亮、trace、计数器 | 把正文当课文背 |
| 多次提问（Prefix 章内） | 加一轮对话 / 加一个请求 | prefill 与 KV 的对比条 | 把多轮和多请求混为一谈 |
| 归因（每章反思阶段内） | 3 张卡分类、提交 | 逐条理由 + 参考总结（先写后看） | 只会复述、不会归因 |

三条硬规则：

- **先表态，再揭晓**：预测题和「请回答」都不默认展开；展开动作由用户触发。
- **不评判人，只对照**：选错只标红并给参考理由，不算分；归因练习才给命中数。
- **留下自己的话**：总结写进 `localStorage`，可复制；参考总结只有点了「对照参考」才出现。

这一层和 P0/P1 的关系：**代码联动、计数器、缓存网格仍然是必要的证据层**，但它们从「主角」降级为「验证用户假设的工具」。

## 3.55 学习闭环：设定 → 前测 → 学习 → 后测 → 反馈（v2.2）

用户需要先知道「这次学的是哪个 LLM、跑在什么 GPU 上」，否则后面的显存/带宽数字没有着力点。

- **本次学习设定**（请回答页顶部）：锚点 LLM Qwen3-Coder-30B-A3B 的规格卡（30.53B / 3.35B 激活、48 层、hidden 2048、32/4 头、128 expert、256K 上下文、96 KiB/token、Q4 18.5 GB）；GPU 可选 RTX 4090/3090/5090、A100 80GB、Apple M2 16GB，展示显存/带宽/BF16 算力/PCIe，并实时给出「Q4 + 32K KV + 运行时 ≈ 23 GB → 能否跑、decode 理论上限、prefill 计算下限」。
- **学习路线图**（请回答页）：5 个阶段（GPU → LLM → KVCache → 推理网络 → 优化）的纵向路线，13 个章节节点直接跳到 `#<chapter>/think`，显存计算器作为工具链接；节点按 `lessons.<ch>.summary` 显示 ✓，阶段显示完成数；放在学习设定之后、前测之前。
- **前测（评分）**：10 题，选完点提交才显示对错和解释；分数存 `localStorage`，作为基线。避免「边做边看答案」。
- **后测（评分）**：同概念换问法的 10 题，提交后生成反馈。
- **问题两层**：每题都有「考什么」一句话概要 + 可展开的「背景 / 怎么想」详细描述，帮助理解题意但不泄露答案。
- **反馈页**：前测/后测分数、变化、按概念对照（提升/维持/退步/仍需复习）、学习完成度（探索步骤数、访问视图数）、错题回看按钮（直达对应章节步骤/视图）；最后的总结也是「概要 + 详细描述」两层。
- 完成度记录：`app.js` 在切换视图/步骤时写 `LK.progress`（`whykvc.progress.v1`）。

## 3.58 三段式章节（v2.4）

**v2.5 收束**：取消独立的「多次提问」「归因与总结」页——多次提问作为 Prefix 章第 11 步内嵌模拟器；三层归因拆成每章 3 张卡，并**与提问反思合并为「反思与总结」阶段**，提交归因后才能完成本章；跨章归因综合放在学习反馈的按概念对照里。

每章不再只是「一组阅读步骤」，而是一个完整学习单元：

| 阶段 | 内容 | 完成条件 |
| --- | --- | --- |
| 请回答 | 3 道预测题（选完才揭示，不计分） | 3 题都想过 → `lessons.<ch>.think` |
| 探索阅读 | 现有 steps：代码 ↔ 讲解 ↔ 可视化 | 记录 `exploreMax` |
| 反思与总结 | 2–4 个开放问题 + 可折叠参考；3 张归因卡；我的总结 → 3 题章节测验 → 参考总结（概要+详细）+ 导出 Markdown | 归因 + 至少 1–2 条反思 + 我的总结 + 章节测验提交后，才能点「完成本章」→ `reflect` + `summary`；测验提交 → `quizScore` |

- 左侧轨道直接显示三阶段与完成状态（请回答/探索阅读/反思与总结），探索阶段下面展开步骤列表；KV 三章的切换放在顶栏「认识 KVCache」菜单，不在左栏重复。非探索阶段仍保留三阶段，隐藏步骤列表、代码 pane / 可视化 / 底部控制条。
- 未完成时显示锁定页：每章状态 + 「去学习」按钮。

## 3.59 章节测验、Markdown 导出与移动代码抽屉（v2.6）

- **章节测验**：每章 `lesson.quiz` 3 题且带 `section`（KV 三章在 `pedagogy.json`，文档章节在各自 JSON 的 lesson），渲染在「我的总结」之后、参考总结之前；提交前不显示对错，提交后显示 `quizScore`、每题解释与「重做章节测验」。状态存 `lessons.<chapter>.quizAnswers / quizScore / quizSubmitted`，学习反馈页汇总各章得分。
- **Markdown 导出**：总结页「导出 Markdown」生成 `whykvc-<chapter>.md`，包含参考总结、我的总结、反思/归因、章节测验和整章内容；文档章节导出时用对应渲染器在游离 DOM 重建教程并转 Markdown，KV 三章直接取 steps 的讲解 HTML。浏览器端用 Blob + `<a download>`，`file://` 也可用。
- **移动代码抽屉**：≤980px 时 KV 代码 pane 默认 `code-collapsed`，只保留 pane header（文件 + 「展开代码 ▾」），点击展开为 42vh 的可滚动代码区；按钮 `aria-expanded` 同步，桌面隐藏。代码区不再固定占 42vh，正文获得更多首屏空间。

## 3.60 可信测评、完成门槛与单位统一（v2.9）

- **测评**：修正后测 b2（8-token prompt + 7 个新 token → 8 + 6 = 14）；前/后测的 `detail` 改写为「背景 / 怎么想」，不出现正确选项原文；选项在题库中打散，避免答案集中在 A/B；每章章节测验扩到 3 题并新增 `section` 字段，覆盖不同小节，删掉与 think 题重复的题目。
- **完成门槛**：`完成本章` 需要归因已提交、至少 2 条反思、自己的总结非空、章节测验已提交（允许低分）；不再只凭归因或空总结拿 ✓。
- **顺序**：反思 → 归因 → 我的总结 → 章节测验 → 参考总结 → 完成/导出；参考总结不再提前出现在测验之前。
- **路线图**：顶部新增「KV 核心路径」（LLM 结构 → KV → Prefix → Paged → 显存计算器，约 3 小时）以及「进阶选修」说明；13 章不再等价推荐。
- **数字与单位**：容量内部统一 byte，展示用十进制 GB/MB；`calc.js`、`train.js`、`gpu_sim.py`、`train_sim.py` 一致；CPU 分桌面（2ch ~80 GB/s）与服务器（8ch ~400 GB/s）两档；PD 公式补齐 attention 项；CUDA 的「两个数量级」改为约 17.8×。`build.py` 的 `check_question_bank` / `check_question_numbers` / `check_sim_units` 拦截回归。
- **测试与交互**：`.github/workflows/ci.yml` 跑 build + `node --check` + `web/tools/smoke.mjs`（19 路由 × 1440/760，active view / console error / 横向溢出）；修复 paged 物理块点击与复用高亮、反馈页 `stage: "summary"` 的跳转。

## 3.6 LLM 结构（v2.1）

**v2.6 补充**：解剖页增加「LLM = 参数 + 结构 + LLM 包 + 运行时状态」四层：
- 「LLM 包里除了权重还有什么」：config / safetensors+index / tokenizer / chat template / generation_config / modeling 代码 / 量化 metadata，标明是否参数与归属；
- 「运行时状态」：常驻（权重、block pool、workspace、CUDA Graph）/ 每请求（block table、KV blocks、seq 状态）/ 每步（hidden、QKV、logits）/ 请求结束后（prefix cache 空闲块）；上下文与并发滑杆实时联动 KV、prefill 激活、常驻杂项与合计；附生命周期表和「LLM 定义 / GPU 提供 / 引擎管理」分工。

**v2.7 / v3 补充**：LLM 结构改为三段式章节（`#anatomy/think|explore|reflect`），探索阶段 10 节：组成、LLM 包、层与维度、token 流水线、GPU 加载、kernel 表、GPU 函数地图、运行时状态、`structure_sim.py` 仿真、自查；左栏使用章节轨道。显存计算器仍是工具，左侧小节导航（`assets/sectionnav.js`）从 `h2.sec-title` 收集 4 个 section，滚动高亮、点击跳转、页面重渲染后重建，折叠到 36px（`whykvc.sectionnav`），窄屏默认收起成顶部条。GPU 函数地图按矩阵计算、归一化/激活、采样、CUDA 运行时、引擎侧分 5 组共 27 条函数；附一次 decode 的 14 步调用链与「常见误区」口径。

用户需要先有「LLM 长什么样」的直觉，否则 KV cache 只是名词。LLM 结构现为三段式章节，探索阶段仍按「先猜 → 证据」组织：

| 小节 | 先问什么 | 给什么证据 |
| --- | --- | --- |
| 组成 | 一个 LM 由哪几块组成？ | embedding → N 层 → norm → lm_head 的 stack，配参数拆分表 |
| 层与维度 | 48 层的第 1 层和第 48 层输出一样宽吗？ | 层网格 + 单层形状 + 可调「层数 / KV 头数」的参数量/KV 计算器 |
| 参数 → token | 参数在生成一步里怎么被用到？ | 8 步流水线：embed → QKV → RoPE → KV 写入 → attention → o_proj → MoE → head，标形状/kernel/谁负责 |
| GPU 加载 | 权重从磁盘到显存经过哪几站？ | 磁盘 → RAM → H2D → 显存四张卡，带体积和 25 GB/s 估算 |
| kernel 表 | 生成本步调用了哪些 GPU 函数？ | 11 行表：阶段 / 操作 / cuBLAS / FlashAttention / reduction，prefill 与 decode 分开 |
| GPU 函数地图 | 同一行 torch 调用落到哪个 kernel？ | 4 层栈 + 5 组 27 条（矩阵 / 归一化 / 采样 / CUDA 运行时 / 引擎）+ 14 步 decode 调用链 + 口径 |
| 自问 | 层数、参数、维度、KV 的关系 | 三张折叠卡，回到「LLM / GPU / 引擎」 |

预设覆盖 TinyModel、Qwen2.5-Coder-7B/32B、Qwen3-Coder-30B-A3B；参数量用公式在构建期算好并断言（30B → 30.53B total / 3.35B active / 96 KiB per token）。

## 3.7 LLM 训练（三段式章节）

LLM 训练是三段式章节（`#train`）：请回答 3 题 → 探索 6 节（含 `example/train_sim.py` 仿真代码）→ 反思与总结：4 题 + 3 张归因卡 + 参考/我的总结 + 章节测验；探索阶段没有代码 pane（训练脚本本身就是内容的一部分），左栏列出 6 个教程小节并随滚动高亮。LLM 结构是三段式章节，显存计算器是顶栏一级入口。主线是「一次 step 的四笔钱」：

- **step**：forward → loss → backward → optimizer；forward 留激活，backward 产梯度，optimizer 维护 m/v + master。
- **显存账本**：权重 `N × bytes`、梯度 `N × bytes`、AdamW 状态 `N × (8 + master)`、激活 `layers × batch × seq × hidden × 12 × bytes`；经验值 AdamW+BF16 约 16 B/参数。
- **交互计算器**：LLM / 精度 / 优化器 / ZeRO 0–3 / DP×TP×PP / 序列 / micro-batch / 激活重算 / offload → 每卡峰值、是否放得下、最大 micro-batch；公式在结果下方可手算核对。
- **并行**：DP 不省显存；ZeRO-1/2/3 逐级切优化器/梯度/参数（FSDP≈ZeRO-3）；TP 切单层矩阵、限 NVLink；PP 切层、有空泡；激活重算用 30% 计算换显存；offload 走 PCIe。
- **可运行脚本**：`example/train_tiny.py`（2 层 TinyGPT、合成循环序列、CPU 一分钟内），演示 forward/loss/backward/step、梯度累积和账本打印；`build.py` 只读源码，不执行。
- **仿真代码**：`example/train_sim.py` 纯标准库模拟每卡账本，打印 DP / ZeRO-1/2/3 / TP / PP / 激活重算下的峰值；页面内嵌源码与 build 期运行出的真实输出。


## 3.8 推理网络（三章三段式章节）

推理网络总览（`#net`）是三段式章节：左侧章节导航有请回答 / 探索阅读 / 反思与总结；探索阶段保留 7 个教程小节（含 `example/net_sim.py` 仿真代码），没有代码 pane，点击小节滚动到对应 `h2`。7 个小节说明「网络在推理里到底做什么」：

- **为什么需要**：TP 每层 all-reduce、MoE all-to-all、PD 的 KV 迁移、prefix cache 块传输；区分「每层小集合通信」与「每请求 KV 迁移」。
- **NCCL**：collectives 语义/流量/步数，ring vs tree vs NVLS/SHARP，NVLink / IB / RoCE / Socket 传输对比。
- **ring 模拟器**：all-reduce / all-gather / reduce-scatter / all-to-all / send-recv，rank 数、消息大小、带宽、延迟可调；逐步显示每个 rank 发哪个 chunk，输出步数、每卡流量、网络总流量、ring 时间、bus bandwidth、tree 对照。
- **RoCE**：RDMA/GPUDirect、RoCEv2 GID、PFC/ECN/DCQCN、MTU、rail-optimized，NCCL 环境变量表、验证命令与常见坑。
- **PD 分离**：router → prefill → KV transfer → decode 的流程；KV 传输 = KV 大小/带宽 + 延迟；计算 TTFT、TPOT、生成总时长、链路请求容量；对比同机不迁移。
- **仿真代码**：`example/net_sim.py` 纯标准库打印 ring 的步数/每卡流量/时间、all-reduce 调度表与 PD 的 KV 传输/TTFT；页面内嵌源码与真实输出。
- **检查清单**：NCCL_DEBUG、all_reduce_perf、show_gids、ibv_devinfo、PD 三段计时。

### 3.8.1 NCCL 学习章节（三段式章节）

`#nccl` 把总览里的「NCCL」一节往下挖成独立学习章节（总览讲「网络在推理里做什么」，这里讲「NCCL 这个库到底怎么搬数据」）。三段式与 `net` 一致：请回答 3 题 → 探索 9 节 → 反思与总结：4 题 + 3 张归因卡 + 参考/我的总结 + 章节测验。

探索 9 节：NCCL 定位（TP / EP / PP / PD 分别落到什么 collective）、六个 collective 的字节/步数/API、**ring all-reduce 逐步播放器**（N / S 可调，显示每步谁把哪个 chunk 发给谁、累计字节、busbw）、tree（上行 reduce + 下行 broadcast、O(log N) 步、带宽换延迟）、**α-β 模型与消息大小扫描**（ring/tree 曲线、crossover、busbw 定义）、拓扑（NVLink / PCIe / IB / RoCE 量级、`nvidia-smi topo -m`、rail-optimized）、测量与排查（nccl-tests、NCCL_DEBUG、协议、常见坑）、仿真代码、自查清单。数据在 `web/tools/nccl.json`，渲染器 `assets/nccl.js` + `assets/nccl.css`，仿真 `example/nccl_sim.py` 在 build 期运行并把 stdout 注入 `simOutput`。

### 3.8.2 CUDA 执行模型（三段式章节，挂在「认识 GPU」下拉）

`#cuda` 回答「GPU 上 kernel 到底怎么跑、为什么 decode 是 memory-bound」。三段式同 `net`：请回答 3 题 → 探索 9 节 → 反思与总结：4 题 + 3 张归因卡 + 参考/我的总结 + 章节测验。

探索 9 节：推理是一串 kernel（decode vs prefill）、执行模型（grid / block / warp / SM、**occupancy 计算器**：block size / regs / shared → 常驻 block、active warps、occupancy、谁先卡住）、内存层级（register / shared / L2 / HBM 的容量-延迟-带宽阶梯、tiling、bank conflict）、**coalescing stride 选择器**（32 线程 × 4 B → sectors / 有效带宽）、**roofline 互动**（batch 滑杆、prefill vs decode 落在哪条线）、**CUDA Graph 对比**（kernel 数 → eager vs replay）、decode 字节账（权重 + KV、量化、paged gather）、仿真代码、自查清单。数据在 `web/tools/cuda.json`，渲染器 `assets/cuda.js` + `assets/cuda.css`，仿真 `example/cuda_sim.py` 在 build 期运行。

### 3.8.3 ECN + DCQCN 无损网络（三段式章节）

`#ecn` 是「推理网络」下拉的第 4 章，把总览里的一页 RoCE 配置展开成可落地的无损网络章节：请回答 3 题 → 探索 10 节 → 反思与总结：4 题 + 3 张归因卡 + 参考/我的总结 + 章节测验。

探索 10 节：
- **为什么需要**：RoCEv2 丢包 → Go-Back-N 重传放大、TTFT/TPOT 抖动；无损 = 不丢包 + 低延迟 + 高吞吐。
- **分层链路**：ECN 两位（Not-ECT/ECT(0)/ECT(1)/CE）→ 交换机 WRED min/max/max_p 标记 → 接收端生成 CNP → 发送端 DCQCN 调速率；PFC 802.1Qbb pause 作为兜底。
- **交换机怎么组网/配置**：leaf-spine、rail-optimized、无阻塞/收敛比、MTU 一致、L3 RoCEv2、PFC 域边界、buffer/headroom 覆盖 pause 反应时间内的在途包、PFC watchdog；trust DSCP、DSCP→TC、每队列 PFC、每队列 ECN WRED、buffer 分配；vendor-neutral 步骤 + Spectrum/Cumulus 与 Arista EOS 示例（命令随版本/SKU 变）。
- **服务器侧配置**：ConnectX `cma_roce_mode -m 2`、`cma_roce_tos`/DSCP、`mlnx_qos --pfc`/ECN、GID index 3、MTU、IRQ 绑核、GPU Direct RDMA、`NCCL_IB_TC` 等；给「网卡/链路 → RoCEv2 → PFC → ECN/DCQCN → 验证」的步骤。
- **如何实现无损**：PFC 兜底防丢 + ECN/DCQCN 细粒度降速；ECN min/max 必须早于 PFC XOFF；只对 lossless 队列开；CNP 高优先级不丢；PFC storm/死锁/HOL 风险与 watchdog。
- **RDMA 业务优先级/PFC**：DSCP→TC→PFC priority 映射示例（RDMA lossless DSCP 26/TC3/PFC 3、CNP 高优先级、存储/管理 best-effort 分队列）；交换机 trust DSCP；PFC 只开必要队列；`NCCL_IB_TC=106` 的对齐。
- **ECN 优先级/阈值**：只标记 ECT/CE、每队列 WRED min/max/max_p、ECN 早于 PFC、标记率过高/过低的影响、CNP 队列与门限；示例参数。
- **排障**：`ethtool -S` 的 PFC/ECN 计数、`mlnx_qos -i`、`nv show qos`、`show interface counters`、`perftest`、PFC watchdog、常见坑。
- **仿真代码**：`example/ecn_sim.py` 由 build 期运行，页面内嵌源码与真实 stdout。
- **自查**：8 条清单 + 6 个折叠问答。

互动：ECN/WRED 阈值滑杆（标记概率曲线 + 队列/标记率/速率收敛）、PFC/ECN 组合场景选择（仅 PFC / ECN+DCQCN / 二者结合）、DSCP→TC→PFC/ECN 映射表（按业务类型看配置）、leaf-spine vs rail-optimized 组网示意。

### 3.9 认识 GPU（下拉：GPU + NPU 与国产算力 + CUDA 执行模型）

「认识 GPU」是独立于「推理网络」的一组章节（`#gpu` GPU + `#npu` NPU 与国产算力 + `#cuda` CUDA 执行模型，顶栏下拉）：把「CUDA / GPU / HCU / 昇腾」放到同一张认知地图上，回答四个问题——CPU 与 GPU 是什么关系、为什么 CPU 在 LLM 领域无法比拟 GPU、为什么 GPU 无法取代 CPU、GPU 的显存访问是否也有 L1/L2 cache。三段式：请回答 3 题 → 探索 10 节 → 反思与总结：4 题 + 3 张归因卡 + 参考/我的总结 + 章节测验。

**NPU 与国产算力**（`#npu`）：把 GPU/NPU/ASIC 的执行模型差异、厂商地图（NVIDIA/AMD/昇腾/寒武纪/海光/壁仞/摩尔线程/沐曦/燧原/昆仑芯/天数）、软件栈（CUDA/ROCm/CANN/Neuware/DTK/MUSA…）和 LLM 选型放进一张地图；探索 8 节 + 厂商筛选 + roofline 仿真。

探索 10 节：
- **CPU/GPU/NPU 分工**：CPU 是 host、GPU/NPU 是 device；CPU 负责 OS/调度/IO/数据搬运，device 负责吞吐型并行；GPU 不是「更强的 CPU」。
- **延迟优化 vs 吞吐优化**：32 核 × 16 lane + 乱序/分支预测/128 MiB LLC vs 132 SM × 128 lane + SIMT + per-SM L1/shared + 50 MiB L2；对比表与「为什么不一样」。
- **LLM 为什么是 GPU 主场**：矩阵乘/attention 的可并行性、算术强度与数据复用；CPU 被 DDR 带宽（~80 GB/s）、向量宽度（512 lane）和功耗卡住；给 decode/prefill 的 CPU vs GPU 倍率（42× / 99×）与 roofline。
- **GPU 为什么取代不了 CPU**：OS/中断/IO/DMA、控制流与分支发散、单线程延迟、小任务 launch（3–8 µs）、Amdahl 上限；结论「CPU 是 host，GPU 是加速器」。
- **显存层级**：HBM / L2 / per-SM L1+shared / 寄存器 / 常量/纹理；和 CPU L1/L2/L3/DDR 的双列容量—延迟—带宽对照。
- **GPU 有没有 L1/L2**：直接回答「有」——L1 与 shared 共用同一块片上 SRAM（carveout 可配）、L2 芯片级共享（H100 ~50 / A100 ~40 / 4090 ~72 MiB）、128 B line、sector 32 B、无 CPU 式 LLC 与跨 socket 目录一致性；并用「连续 vs 跨步访问」互动展示 line/sector 利用率。
- **访存模式**：coalescing、latency hiding、memory-bound，和 CPU cache 优化的差异。
- **生态**：CUDA（NVIDIA 软件栈，不是芯片）、ROCm+HIP（AMD）、CANN+AscendCL（昇腾）、HCU=DTK+HIP（海光 DCU，HAMi 设备名）、OpenCL/oneAPI/Triton；vLLM/SGLang 的 backend 与 custom ops 对接路径。
- **仿真代码**：`example/gpu_sim.py` 由 build 期运行，页面内嵌源码与真实 stdout。
- **自查**：8 条清单 + 6 个折叠问答（含「GPU 有 L1/L2 cache 吗」）。

互动：CPU vs GPU 对比条（指标 × 档位 H100/A100/4090/昇腾 910B/HCU）、统一存储层级阶梯（点击看容量/延迟/带宽/是否 cache，一键高亮 GPU 的 L1/L2）、场景分工选择器（6 类任务 → CPU/GPU 主导）、coalescing/cache-line 路径（stride 1 vs 8）。

### 3.9.5 Transformer 与 LLM（认识 LLM 下拉）

`#transformer` 回答「LLM 从哪来、为什么是它胜出」：RNN/LSTM 的串行与长距离限制 → 自注意力 → Transformer 块与三种架构 → 2017 至今的时间线 → 缩放定律 → 为什么 LLM 成为通用入口（统一 token 接口、自监督数据、上下文学习、硬件友好、缩放经济学、生态飞轮）→ 边界（幻觉、O(n²)、成本、专用模型仍最优）。探索 9 节 + 序列长度滑杆（注意力 N² / KV 线性）+ `example/transformer_sim.py` 账本；三段式：请回答 → 探索阅读 → 反思与总结。

## 3.10 LLM 优化（下拉）

「LLM 优化」是独立于前四组的一章（`#opt`，顶栏下拉：总览 / FlashAttention / LLM 量化 / 投机解码锚点）：把「算子优化 / FlashAttention / PagedAttention / LLM 量化 / 其他方向」放到同一张优化地图上。三段式：请回答 3 题 → 探索 11 节 → 反思与总结：4 题 + 3 张归因卡 + 参考/我的总结 + 章节测验。

探索 11 节：
- **先找瓶颈**：算力/带宽/容量/延迟/吞吐/通信六约束；优化不是堆技巧。
- **优化地图**：算子/kernel、显存与缓存、数值精度、解码与调度、并行与通信、LLM 结构六类，各自攻击什么、代价是什么。
- **FlashAttention**：IO-aware tiling + online softmax，避免 N×N 物化、HBM 流量 O(N²)→O(N)；prefill 收益大、decode 收益小（FlashDecoding/split-K）；HBM 流量计算器。
- **PagedAttention**：作为显存管理优化（block table/gather/碎片/跨请求共享），是引擎层而不是单个 kernel，引用 KV 章。
- **LLM 量化**：PTQ/QAT、W4A16（GPTQ/AWQ/GGUF）、W8A8（SmoothQuant）、FP8 E4M3/E5M2 与 scaling 粒度、KV 量化、硬件/kernel 支持；量化计算器。
- **投机解码**：draft + verify、接受率 α、树验证、加速比公式与下界、multi-token prediction；加速比滑杆。
- **调度与系统**：continuous batching、chunked prefill、prefix cache、PD 分离、CUDA Graph、kernel fusion、TP/PP/EP、overlap/offload。
- **LLM 结构**：GQA/MQA、MLA、sliding window/attention sinks、MoE、linear/hybrid attention 的取舍。
- **选型**：workload → 瓶颈 → 2–3 个高杠杆技术；避免重复/冲突；先测量（ncu/nsys/benchmark）、A/B。
- **仿真代码**：`example/opt_sim.py` 由 build 期运行并内嵌真实 stdout。
- **自查**：8 条清单 + 6 个折叠问答。

互动：瓶颈→方向选择器、FlashAttention HBM 流量对比、量化计算器（LLM/权重 bits/KV bits → 显存与 decode 加速比）、投机解码加速比滑杆。

### 3.11 文档型章节（net / train / nccl / cuda / ecn / gpu / opt）

`net`/`train`/`nccl`/`cuda`/`ecn`/`gpu`/`opt` 不套代码章节的三栏 DOM，而是 `.chapter.doc-chapter` 两栏：左侧章节导航（三阶段 + 教程小节），右侧 `.doc-main > .workshop` 整篇教程。

- 三阶段复用 `workshop.js` 的 `renderStageThink` 与 `renderStageReview`，lesson 存在各章 JSON（`net.json` / `train.json` / `nccl.json` / `cuda.json` / `ecn.json` / `gpu.json` / `opt.json`）的 `lesson` 字段，`build.py` 合并进 `data.lessons`。
- 探索阶段渲染整篇教程；左栏步骤来自教程的 `h2.sec-title`（net 7 / train 6 / nccl 9 / cuda 9 / ecn 10 / gpu 10 / opt 11 节），点击滚到对应 `h2`，滚动时用 `currentDocIndex` 同步 active、计数和 `exploreMax`；URL 用 `#<chapter>/N` 记录当前小节。
- 没有代码 pane / 可视化 pane / 底部 timeline；切换 stage 时右侧内容整体替换，和 KV 章的三段式节奏一致。
- 各章的「仿真代码」小节把 `example/<chapter>_sim.py` 用 codeview 嵌入，并内联 `build.py` 运行出的真实 stdout；纯标准库，不依赖 NCCL / torch。
- `stop`：离开任一文档章节或切出 explore 时调用对应渲染器的 `stop()`，停掉 NCCL 模拟播放与定时器（`DOC_MODULES` 统一分发）。

## 4. 页面布局

```
┌──────────────────────────────────────────────────────────────────────┐
│ whykvc · 启发式阅读  请回答  认识GPU▾  认识KVCache▾  推理网络▾  认识 LLM▾  LLM 优化▾  学习反馈  [暗/亮] │
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

顶栏：五个学习章节下拉——「认识 GPU」（GPU / NPU 与国产算力 / CUDA 执行模型）、「认识 LLM」（Transformer 与 LLM / LLM 结构 / LLM 训练）、「认识 KVCache」（三个 KV 子章）、「推理网络」（总览 / NCCL 学习章节 / ECN + DCQCN）、「LLM 优化」（总览 / FlashAttention / LLM 量化 / 投机解码）；一级入口「请回答 · 前测」「显存计算器」「学习反馈」。触发按钮会把当前所在项写进标签（如「认识 KVCache · KV」「LLM 训练 · 训练教程」），并给菜单项加 `aria-current`；打开/关闭用 `aria-expanded` 同步，选中、点外部、Esc 都关闭；菜单宽度在窄屏用 JS 夹在视口内，导航整体换行而不是横向滚动。

移动端（< 980px）：左侧导航默认收起成 38px 轨道；展开后章节排成 3 列、三阶段排成 3 列（隐藏进度文字、保留完成打勾），步骤列表限高滚动；KV 代码区默认折叠成抽屉（header 45px，展开 42vh），可视化与讲解上下堆叠；`net`/`train` 是导航在上、教程内容占剩余高度（`auto minmax(0,1fr)`），底部无 timeline。

KV 三章：桌面是 200px 章节导航 + 代码 + 讲解/可视化三栏；`transformer`/`anatomy`/`net`/`train`/`nccl`/`cuda`/`ecn`/`gpu`/`npu`/`opt`：200px 章节导航 + 整篇教程两栏（无代码 pane）；显存计算器仍是工具，200px 小节导航 + 正文。导航 sticky 且可折叠到 36px（同宽、同底色、同折叠按钮）；≤980px 默认收起成正文顶部条，展开后列表限高滚动，正文表格可横向滚动。

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

- 输入：LLM preset（Qwen3-Coder-30B-A3B / Qwen2.5-Coder-7B/14B/32B / Qwen3.5-4B/9B）、量化、上下文长度、KV dtype、并发数、引擎。
- 输出：`权重 + KV + 激活 + 运行时` 的堆叠条，以及 24/32/48 GB 卡的结论；prefill 的 FLOPs 拆成 `GEMM` 与 `full attention` 两条腿，附估算耗时。
- 公式与 `macos/ollama/README.md` 的表格一致，悬停显示推导；preset 参数硬编码在 `data/presets.js`，便于和文档同步。

### 5.6（可选）LLM 结构

- 横向流程图：`RMSNorm → q/k/v + q_norm/k_norm → RoPE → KV 写入 → GQA → o_proj → 残差 → RMSNorm → router → top-8 experts → 合并 → 残差`。
- 每个节点可点开：形状（n=10K / n=1 切换）、FLOPs、以及对应的源码/文档出处。
- prefill/decode 切换时，张量形状从 `[n, d]` 变成 `[1, d]`，直观看出「大 GEMM → GEMV」。

## 6. 代码 ↔ 文档 ↔ 状态 同步（核心机制）

### 6.1 step LLM

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
├── index.html             # 页面壳（无框架、无 CDN；静态资源带 ?v=YYYY-MM-DD）
├── DESIGN.md
├── README.md
├── assets/
│   ├── style.css          # 暗色默认 + 亮色变量 + 响应式
│   ├── app.js             # hash 路由、章节/文档章节装配、stage 与滚动联动
│   ├── codeview.js        # 自写 Python tokenizer、行号、行高亮、行链接
│   ├── workshop.js        # 请回答 / 归因 / 总结 / 反馈 / 作业 / 进度
│   ├── anatomy.js / calc.js / sectionnav.js
│   ├── net.js             # 推理网络总览：NCCL 模拟 / RoCE / PD 计算器
│   ├── nccl.js / nccl.css # NCCL 学习章节
│   ├── cuda.js / cuda.css # CUDA 执行模型
│   ├── ecn.js / ecn.css   # ECN + DCQCN 无损网络
│   ├── gpu.js / gpu.css   # 认识 GPU
│   ├── npu.js / npu.css   # NPU 与国产算力
│   ├── transformer.js / transformer.css  # Transformer 与 LLM
│   ├── opt.js / opt.css   # LLM 优化
│   ├── train.js           # LLM 训练章节
│   └── viz/               # KV 三章的代码步骤可视化（kv.js / prefix.js / paged.js）
├── data/
│   ├── data.js            # 生成：代码 + 文档 HTML + steps + lessons（提交）
│   └── traces.js          # 生成：第 1/2/3 章状态快照 + reuse（提交）
└── tools/
    ├── build.py           # data.js / traces.js 的唯一生成入口
    ├── collect_traces.py  # KV 三章 trace + reuse 账本
    ├── steps.json / steps.prefix.json / steps.paged.json
    ├── pedagogy.json / anatomy.json
    ├── net.json / train.json
    └── nccl.json / cuda.json / ecn.json / gpu.json / opt.json
```

- **无框架**：原生 DOM + SVG + CSS；可视化组件挂到 `window.LK.viz.<name>`，由 app.js 按 `step.viz` 调用。
- **无模块加载器**：`<script>` 顺序加载（`data.js` → `traces.js` → `viz/*.js` → `app.js`），保证 `file://` 可用；需要 `fetch` 时再提供 `python3 -m http.server` 的说明。
- **语法高亮**：自写约 80 行 Python tokenizer（关键字/字符串/注释/数字/装饰器/函数名），避免引入 highlight.js；颜色走 CSS 变量，与暗/亮主题一致。
- 图表用 SVG/DOM，动画用 CSS transition + `requestAnimationFrame`。

## 9. 视觉与可用性

- **配色语义固定**：prompt token、生成 token、命中缓存、新分配块、被淘汰块各一套颜色；色盲安全模式下加纹理/形状区分。
- **字体**：代码等宽（系统 `ui-monospace`），正文系统无衬线；行高 1.5–1.6。
- **动效**：200–300ms；`prefers-reduced-motion: reduce` 时直接跳变。
- **无障碍**：`aria-live="polite"` 播报计数器；步骤导航项与折叠按钮都是原生 `<button>`（折叠带 `aria-expanded`）；顶栏下拉带 `aria-haspopup="menu"`/`aria-expanded`/`aria-current`，支持 ↑/↓/Home/End/Esc/Tab 与点外部关闭；焦点环可见；对比度 ≥ 4.5:1。
- **主题**：暗色默认，可切亮色，选择存 `localStorage`。
- **打印**：`@media print` 隐藏控件，输出「讲解 + 代码 + 当前状态」。
- **深链**：`#kv/3`、`#net/4`、`#net/reflect`、`#nccl/3`、`#cuda/5`、`#cuda/reflect`、`#ecn/3`、`#ecn/6`、`#gpu/6`、`#opt/3`、`#opt/reflect`、`#anatomy/3`、`#anatomy/reflect`、`#calc`。

## 10. 实施计划

| 阶段 | 内容 | 产出 |
| --- | --- | --- |
| P0 | `build.py` + 页面壳 + 代码高亮 + 第 1 章文档联动（无动画） | 能双击打开、左右联动 |
| P1 | 第 1 章：双栏播放、计数器、缓存网格、滑动窗口/reorder/fork | 第 1 章完整 |
| P1.5 | 思考层：请回答 10 题、每步请回答、多次提问模拟器、归因 + 总结 | 主流程改为「请回答 → 证据 → 归因」 |
| P1.6 | LLM 结构：组成/层与维度/参数→token/GPU 加载/kernel 表 + 4 个预设 | 补上「LLM 本身」的直觉 |
| P2 ✅ | 第 2 章：prefix 场景 + 块池 + LRU 动画 | 已完成 |
| P3 ✅ | 第 3 章：block table + append/gather 动画 + 碎片对比 | 已完成 |
| P4 ✅ | 计算器 + LLM 结构 + 亮色主题 + 响应式 + 无障碍 | 已完成（键盘/跳转链接/aria-live/主题记忆） |
| P5 ✅ | README 入口、GitHub Pages 部署（`.github/workflows/pages.yml`） | 已完成，待仓库开启 Pages |
| P6 ✅ | 推理网络总览 + LLM 训练三段式章节（`net.json` / `train.json` + 渲染器 + 仿真代码） | 已完成 |
| P7 ✅ | 推理网络深造：NCCL 学习章节 + CUDA 执行模型（`nccl.json` / `cuda.json` + 渲染器 + `nccl_sim.py` / `cuda_sim.py`） | 已完成：两章各 9 节、3+ 互动、三段式可完成 |
| P8 ✅ | 认识 GPU（`gpu.json` + `gpu.js` + `gpu_sim.py`）：CPU/GPU/NPU 分工、LLM 适配、显存与 L1/L2、CUDA/ROCm/昇腾/HCU 生态 | 已完成：10 节、4 个互动、三段式可完成 |
| P9 ✅ | ECN + DCQCN 无损网络（`ecn.json` + `ecn.js` + `ecn_sim.py`）：ECN/WRED、CNP/DCQCN、交换机/服务器配置、组网、PFC、优先级与阈值 | 已完成：10 节、4 个互动、三段式可完成 |
| P10 ✅ | LLM 优化（`opt.json` + `opt.js` + `opt_sim.py`）：瓶颈地图、FlashAttention、PagedAttention、量化、投机解码、调度与结构、选型 | 已完成：11 节、4 个互动、三段式可完成 |
| P11 ✅ | 章节测验（每章 3 题计分）+ Markdown 导出（总结/整章）+ 移动端代码抽屉 | 已完成：测验计分与反馈汇总、导出 `.md`、980px 以下抽屉 |
| P12 ✅ | NPU 与国产算力（`npu.json` + `npu.js` + `npu_sim.py`）：GPU/NPU/ASIC、厂商地图、软件栈、选型 | 已完成：8 节、厂商筛选、三段式可完成 |
| P13 ✅ | Transformer 与 LLM（`transformer.json` + `transformer.js` + `transformer_sim.py`）：自注意力、架构、缩放、为什么胜出 | 已完成：9 节、序列滑杆、三段式可完成 |
| P14 ✅ | LLM 结构转为三段式章节（`anatomy.json` + `anatomy.js` + `structure_sim.py`）：参数/层/维度/GPU 加载/kernel 地图/运行时 + 10 节仿真 | 已完成：三段式、10 节、仿真输出、章节测验 |

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
5. 960px 以下布局可用，键盘可完成全部步进操作；KV 代码抽屉可展开/收起。
6. 每章总结页的章节测验可提交、计分、重做；学习反馈显示各章得分。
7. 「导出 Markdown」下载的文件包含参考总结、我的总结、反思/归因、测验与整章内容。
