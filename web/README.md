# learnkvc 启发式阅读器（v2.8）

把第 1 章的阅读导图和源码放在同一个页面里，但主线不是「读文档」，而是**先回答、再看证据、最后归因和总结**：

```
请回答（6 个预测题）
→ 认识 GPU（总览 → CUDA 执行模型，两章四段式）
→ 认识 KVCache（三章四段式；多次提问在 Prefix 章第 11 步）
→ 推理网络（总览 → NCCL 学习章节 → ECN + DCQCN，三章四段式）
→ 认识模型（模型解剖 / 模型训练 / 显存计算器）
→ LLM 优化（FlashAttention / 量化 / 投机解码 / 调度与结构优化 / 选型，四段式）
→ 后测 → 学习反馈
```

知识不是目标；目标是让你先形成假设，再用代码和 trace 修正它，最后能用自己的话解释「一次提问、多次提问、模型 / GPU / 引擎」。

## 打开

GitHub Pages（需在仓库 Settings → Pages 选择 GitHub Actions）：<https://newllm.github.io/learnkvc/web/index.html>。


直接双击 `web/index.html`（数据已内联在 `data/*.js`，`file://` 可用，无需服务器、无需联网）。

也可以起一个本地静态服务：

```bash
python3 -m http.server 8000
# http://127.0.0.1:8000/web/index.html
```

## 视图

| 视图 | 做什么 |
| --- | --- |
| 请回答（前测） | 先说明本次学习设定：锚点模型 Qwen3-Coder-30B-A3B + 可选 GPU（4090/3090/5090/A100/M2）的显存、带宽、算力，并列出本次覆盖（认识 GPU / KV 三章 / 推理网络 / 认识模型 / LLM 优化）；10 题前测**评分**（提交后才显示对错），覆盖模型构成 / 一次提问 / 多次提问 / 分页 / 运行时状态 / 三层分工 / GPU 与 CPU / 集合通信 / 无损网络 / 优化技术，每题带概要和详情 |
| 认识 GPU · 总览 | 四段式章节：请回答 3 题 → 探索 10 节（CPU/GPU/NPU 分工、CPU vs GPU 延迟/吞吐、LLM 为什么适合 GPU、GPU 为什么取代不了 CPU、HBM/L2/L1+shared 显存层级、**GPU 有没有 L1/L2 cache** + 对比 CPU、coalescing/latency hiding、CUDA/ROCm/CANN 昇腾/HCU 生态、仿真代码、自查）→ 反思 4 题 + 3 张归因卡 → 总结 |
| 认识 GPU · CUDA 执行模型 | 四段式章节：请回答 3 题 → 探索 9 节（kernel 序列、grid/block/warp/SM + **occupancy 计算器**、内存层级、**coalescing stride 选择器**、**roofline 互动**、**CUDA Graph 对比**、decode 字节账、仿真代码、自查）→ 反思 4 题 + 3 张归因卡 → 总结 |
| ① KV cache 章 | **四段式**（请回答 → 探索 → 反思与归因 → 总结归纳）：14 步代码高亮 + 双栏生成播放 + 缓存网格 + 掩码；请回答 3 题预测、4 个反思问题 + 3 张归因卡、总结卡 |
| ② Prefix cache 章 | **四段式**：15 步探索 + 三场景（共享前缀 / 池子不足 LRU / 相同 token 不同位置）请求时间线、块池、哈希链、引用计数与淘汰；第 11 步内嵌「多次提问」多轮/多请求模拟器 |
| ③ PagedAttention 章 | **四段式**：16 步 + 逻辑 token → block table → 物理块池，可播放 append/free/reuse，块内 slot 与内部碎片；请回答/反思/总结 |
| 归因 | 不单独成页：并入每章「反思与归因」阶段（3 张卡）；总结归纳只保留总结卡；跨章综合在学习反馈的按概念对照 |
| 模型解剖（认识模型菜单） | 左侧 9 节导航（滚动高亮/点击跳转）：4 个预设（TinyModel / 7B / 32B / 30B-A3B）的参数/结构/模型包/运行时四层构成、模型包清单（config/tokenizer/模板/生成配置）、层数与维度计算器、8 步 token 流水线、GPU 加载流程、11 类 kernel、**GPU 函数地图（PyTorch → cuBLASLt / CUTLASS / FlashAttention / FlashInfer / NCCL → CUDA kernel → 硬件单元，5 组 27 条 + 14 步 decode 调用链 + 口径）**、运行时状态地图（常驻/每请求/每步，KV 与上下文×并发联动） |
| 显存计算器（认识模型菜单） | 左侧 4 节导航：模型/量化/KV 精度/上下文/并发/GPU 六个输入，输出权重+KV+运行时、是否放得下、prefill 计算下限、decode 带宽上限 |
| 推理网络 · 总览 | 四段式章节：请回答 3 题 → 探索 7 节（推理为何需要网络、NCCL、**ring 模拟器**、RoCE/RDMA、**PD 分离计算器**、**Python 仿真代码**、网络账本）→ 反思 4 题 + 3 张归因卡 → 总结；左侧章节导航 + 分节跳转 |
| 推理网络 · NCCL 学习章节 | 四段式章节：请回答 3 题 → 探索 9 节（NCCL 定位、六个 collective、**ring 逐步播放器**、tree、**α-β 扫描与 crossover**、拓扑、测量/排查、**Python 仿真代码**、自查）→ 反思 4 题 + 3 张归因卡 → 总结；`example/nccl_sim.py` 在 build 期运行并内嵌真实输出 |
| 推理网络 · ECN + DCQCN | 四段式章节：请回答 3 题 → 探索 10 节（ECN/WRED 标记、CNP 与 DCQCN 闭环、**交换机组网与配置**、**服务器侧配置**、PFC 无损与风险、**RDMA 业务优先级/PFC**、**ECN 优先级与阈值**、排障、仿真代码、自查）→ 反思 4 题 + 3 张归因卡 → 总结 |
| 模型训练 | 四段式章节：请回答 3 题 → 探索 6 节（训练 step、显存账本 + 计算器、并行、最小脚本、**Python 仿真代码**、检查清单）→ 反思 4 题 + 3 张归因卡 → 总结；模型解剖/显存计算器收在「认识模型」下拉里 |
| LLM 优化 · 总览 | 四段式章节：请回答 3 题 → 探索 11 节（先找瓶颈、优化地图、**FlashAttention HBM 计算器**、PagedAttention、**量化计算器**、**投机解码加速比**、调度与系统、模型结构、选型决策、仿真代码、自查）→ 反思 4 题 + 3 张归因卡 → 总结 |
| ⑤ 学习反馈 | 学完再做 10 题**后测**（同概念换问法，同样带概要 + 详情，覆盖认识 GPU / KV 三章 / 推理网络 / 认识模型 / LLM 优化）：前测/后测分数、按概念对照（提升/维持/退步）、学习完成度（107 步 / 12 个学习视图）、错题回看链接（可直达章节/解剖页）；总结给「概要 + 详细描述」两层 |

另有：顶栏五个学习内容下拉——「认识 GPU」下拉（总览 / CUDA 执行模型）、「认识 KVCache」下拉（单请求 / Prefix / Paged）、「推理网络」下拉（总览 / NCCL 学习章节 / ECN + DCQCN）、「认识模型」下拉（模型解剖 / 模型训练 / 显存计算器）、「LLM 优化」下拉（总览 / FlashAttention / 模型量化 / 投机解码锚点）；触发按钮显示当前所在项（如「认识 KVCache · KV」「推理网络 · NCCL」「模型训练 · 训练教程」），支持 ↑/↓/Esc/点外部关闭，窄屏导航换行不横向溢出。章节导航统一在**左侧轨道、可收缩**——顶部是「章节」切换（三章标题，当前高亮、完成打勾；折叠时只留 ①②③）与四个阶段（请回答/探索/反思/总结），探索阶段展开步骤列表；折叠状态记忆、窄屏默认收起；非探索阶段只保留章节（如有）与阶段列表。认识 GPU/认识 KVCache/推理网络/认识模型/LLM 优化使用**章节四段式左栏**（四阶段 + 10/9/7/9/10/6/11 个教程小节，点击滚动、滚动高亮）；模型解剖/显存计算器仍是工具，另有一条**小节导航**（与章节导航同款：200px 可折叠左栏，窄屏默认收起成顶部条），由 `h2` 自动生成并在换预设/拉滑杆后重建。亮/暗主题切换、键盘导航（←/→/空格）和 `file://` 离线支持。设计见 [`DESIGN.md`](DESIGN.md)。

## 改内容后重新生成

页面里的代码、讲解、步骤、题目、trace 都从仓库文件生成，不要在 `data/*.js` 里手改：

```bash
python3 web/tools/build.py
```

- 静态资源带 `?v=YYYY-MM-DD`（见 `web/index.html`）；改了 `assets/` 或 `data/` 后同步更新版本号，避免 GitHub Pages 缓存混用新旧 JS/HTML（曾因此点出 `#undefined`）。
- 代码：`example/kv_cache.py`（`ast` 解析符号区间）
- 讲解：`example/KV_CACHE_READING.md`（按 `##` / `###` 切节）
- 步骤：`web/tools/steps.json`
- 思考层：`web/tools/pedagogy.json`（预测题、每步 question、归因卡片、反思提示）
- 模型解剖：`web/tools/anatomy.json`（预设架构、8 步生成流水线、GPU 加载阶段、kernel 表、GPU 函数地图 `gpuFunctions`）；参数量/KV 由 `build.py` 按公式计算并断言
- 推理网络 · 总览：`web/tools/net.json`（NCCL/RoCE/PD 文案与表格；章节 lesson 在同文件的 `lesson` 字段；模拟器与计算器在 `assets/net.js`；纯 Python 仿真 `example/net_sim.py`，build 期运行）
- 推理网络 · NCCL 学习章节：`web/tools/nccl.json`（九节教程 + lesson；渲染器 `assets/nccl.js` + `assets/nccl.css`；仿真 `example/nccl_sim.py`，build 期运行并注入 stdout）
- 推理网络 · ECN + DCQCN：`web/tools/ecn.json`（十节教程 + lesson；渲染器 `assets/ecn.js` + `assets/ecn.css`；仿真 `example/ecn_sim.py`，build 期运行并注入 stdout）
- 认识 GPU · 总览：`web/tools/gpu.json`（十节教程 + lesson；渲染器 `assets/gpu.js` + `assets/gpu.css`；仿真 `example/gpu_sim.py`，build 期运行并注入 stdout）
- 认识 GPU · CUDA 执行模型：`web/tools/cuda.json`（九节教程 + lesson；渲染器 `assets/cuda.js` + `assets/cuda.css`；仿真 `example/cuda_sim.py`，build 期运行并注入 stdout）
- LLM 优化：`web/tools/opt.json`（十一节教程 + lesson；渲染器 `assets/opt.js` + `assets/opt.css`；仿真 `example/opt_sim.py`，build 期运行并注入 stdout）
- 模型训练：`web/tools/train.json`（章节 lesson 在同文件的 `lesson` 字段）（步骤/账本/并行/清单）+ `example/train_tiny.py`（build.py 只读源码，不执行）+ `example/train_sim.py`（build 期运行，注入真实输出）
- 学习闭环：前测/后测分数、所选 GPU、访问过的视图/步骤都存 `localStorage`（key `learnkvc.progress.v1`），
  学习反馈页据此生成前后对照；视图/步骤的访问由 `app.js` 在切换时通过 `LK.progress` 记录
- 页面回归（可选）：`for f in web/assets/*.js web/assets/viz/*.js; do node --check "$f" || exit 1; done`
- 状态：`web/tools/collect_traces.py`
  - 装了 PyTorch：跑真实 `generate()` 得到 tokens / logits / 计数
  - 没装：用 README 记录的 token 序列 + `print_summary()` 的断言公式生成解析 trace（文件里会标注 `source: analytic`）
  - 另外生成「多轮 / 多请求」的复用账本（`reuse` 字段），供 Prefix 章第 11 步「多次提问」使用

## 文件

```
web/
├── index.html          # 页面壳（无框架、无 CDN）
├── DESIGN.md           # 设计稿（信息架构、四段式机制、P0–P10）
├── assets/
│   ├── style.css
│   ├── app.js          # hash 路由、十个学习章节（含 7 个文档章节）+ 工具视图、步骤/滚动联动
│   ├── codeview.js     # Python 高亮 + 行号 + 代码区滚动
│   ├── workshop.js     # 请回答 / 归因 / 总结 / 后测反馈 / 学习进度
│   ├── anatomy.js      # 模型组成 / 层与维度 / 参数→token / GPU 加载 / kernel 表
│   ├── train.js        # 训练章节：账本 / 计算器 / 并行 / 训练与仿真代码
│   ├── net.js          # 推理网络总览：NCCL 模拟 / RoCE / PD 计算器 / 仿真代码
│   ├── nccl.js         # NCCL 学习章节：逐步播放器 / α-β 扫描 / 拓扑 / 仿真代码
│   ├── cuda.js         # CUDA 执行模型：occupancy / coalescing / roofline / graph / 字节账
│   ├── gpu.js          # 认识 GPU：CPU/GPU 对比、存储层级、L1/L2、生态、仿真代码
│   ├── ecn.js          # ECN + DCQCN：WRED 阈值、PFC/DCQCN、DSCP→TC、组网、仿真代码
│   ├── opt.js          # LLM 优化：瓶颈地图、FlashAttention/量化/投机解码计算器、选型
│   ├── nccl.css / cuda.css / gpu.css / ecn.css / opt.css  # 深造章节的专属样式
│   ├── sectionnav.js   # 模型解剖/显存计算器的小节导航
│   └── viz/kv.js       # 第 1 章的可视化组件
├── data/               # 生成物（已生成，双击即用）
│   ├── data.js         # 代码 + 讲解 HTML + steps + 思考层内容
│   └── traces.js       # 生成状态快照 + reuse 账本
└── tools/
    ├── build.py
    ├── collect_traces.py
    ├── steps.json
    ├── pedagogy.json
    ├── net.json / train.json
    ├── nccl.json / cuda.json / ecn.json / gpu.json / opt.json
    └── anatomy.json
```
