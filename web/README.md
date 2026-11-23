# learnkvc 思考式阅读器（v2.5）

把第 1 章的阅读导图和源码放在同一个页面里，但主线不是「读文档」，而是**先回答、再看证据、最后归因和总结**：

```
请回答（6 个预测题） → ① KV cache 章 → ② Prefix cache 章 → ③ PagedAttention 章
  （每章：请回答 → 探索阅读 → 反思与归因 → 总结归纳；多次提问在 Prefix 章第 11 步）
→ 认识 KVCache（三章） → 推理网络（四段式） → 模型训练（四段式 + 模型解剖/显存计算器工具） → 后测 → 学习反馈 → 课后作业
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
| 请回答（前测） | 先说明本次学习设定：锚点模型 Qwen3-Coder-30B-A3B + 可选 GPU（4090/3090/5090/A100/M2）的显存、带宽、算力，并列出本次覆盖（三章 + 模型解剖 + 计算器）；6 题前测**评分**（提交后才显示对错），覆盖模型构成 / 一次提问 / 多次提问 / 分页 / 运行时状态 / 三层分工，每题带概要和详情 |
| ① KV cache 章 | **四段式**（请回答 → 探索 → 反思与归因 → 总结归纳）：14 步代码高亮 + 双栏生成播放 + 缓存网格 + 掩码；请回答 3 题预测、4 个反思问题 + 3 张归因卡、总结卡 |
| ② Prefix cache 章 | **四段式**：16 步探索 + 三场景（共享前缀 / 池子不足 LRU / 相同 token 不同位置）请求时间线、块池、哈希链、引用计数与淘汰；第 11 步内嵌「多次提问」多轮/多请求模拟器 |
| ③ PagedAttention 章 | **四段式**：15 步 + 逻辑 token → block table → 物理块池，可播放 append/free/reuse，块内 slot 与内部碎片；请回答/反思/总结 |
| 归因 | 不单独成页：并入每章「反思与归因」阶段（3 张卡）；总结归纳只保留总结卡；跨章综合在三章对照矩阵 |
| 模型解剖（模型训练菜单） | 左侧 9 节导航（滚动高亮/点击跳转）：4 个预设（TinyModel / 7B / 32B / 30B-A3B）的参数/结构/模型包/运行时四层构成、模型包清单（config/tokenizer/模板/生成配置）、层数与维度计算器、8 步 token 流水线、GPU 加载流程、11 类 kernel、**GPU 函数地图（PyTorch → cuBLASLt / CUTLASS / FlashAttention / FlashInfer / NCCL → CUDA kernel → 硬件单元，5 组 27 条 + 14 步 decode 调用链 + 口径）**、运行时状态地图（常驻/每请求/每步，KV 与上下文×并发联动） |
| 显存计算器（模型训练菜单） | 左侧 4 节导航：模型/量化/KV 精度/上下文/并发/GPU 六个输入，输出权重+KV+运行时、是否放得下、prefill 计算下限、decode 带宽上限 |
| 推理网络 | 四段式章节：请回答 3 题 → 探索 7 节（推理为何需要网络、NCCL、**ring 模拟器**、RoCE/RDMA、**PD 分离计算器**、**Python 仿真代码**、网络账本）→ 反思 4 题 + 3 张归因卡 → 总结；左侧章节导航 + 分节跳转 |
| 模型训练 | 四段式章节：请回答 3 题 → 探索 6 节（训练 step、显存账本 + 计算器、并行、最小脚本、**Python 仿真代码**、检查清单）→ 反思 4 题 + 3 张归因卡 → 总结；模型解剖/显存计算器收在「模型训练」下拉里 |
| ⑤ 学习反馈 | 学完再做 6 题**后测**（同概念换问法，同样带概要 + 详情，覆盖三章 + 模型解剖）：前测/后测分数、按概念对照（提升/维持/退步）、学习完成度（58 步 / 7 个学习视图）、错题回看链接（可直达章节/解剖页）；总结给「概要 + 详细描述」两层 |
| ⑥ 课后作业 | 三章总结归纳完成后解锁：三章对照矩阵 + 模型解剖对照矩阵 + 6 道题（其他领域迁移 / KV 公式 / TTL 设计 / 自由类比 / 模型包体检 / 运行时账本）+ 参考答案 + 提交记录 |

另有：顶栏三个学习入口——「认识 KVCache」下拉（单请求 / Prefix / Paged）、「推理网络」直接入口、「模型训练」下拉（模型训练 / 模型解剖 / 显存计算器）；触发按钮显示当前所在项（如「认识 KVCache · KV」「模型训练 · 训练教程」），支持 ↑/↓/Esc/点外部关闭，窄屏导航换行不横向溢出。章节导航统一在**左侧轨道、可收缩**——顶部是「章节」切换（三章标题，当前高亮、完成打勾；折叠时只留 ①②③）与四个阶段（请回答/探索/反思/总结），探索阶段展开步骤列表；折叠状态记忆、窄屏默认收起；非探索阶段只保留章节（如有）与阶段列表。推理网络/模型训练使用**章节四段式左栏**（四阶段 + 7/6 个教程小节，点击滚动、滚动高亮）；模型解剖/显存计算器仍是工具，另有一条**小节导航**（与章节导航同款：200px 可折叠左栏，窄屏默认收起成顶部条），由 `h2` 自动生成并在换预设/拉滑杆后重建。亮/暗主题切换、键盘导航（←/→/空格）和 `file://` 离线支持。设计见 [`DESIGN.md`](DESIGN.md)。

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
- 推理网络：`web/tools/net.json`（NCCL/RoCE/PD 文案与表格；章节 lesson 在同文件的 `lesson` 字段；模拟器与计算器在 `assets/net.js`；纯 Python 仿真 `example/net_sim.py`，build 期运行）
- 模型训练：`web/tools/train.json`（章节 lesson 在同文件的 `lesson` 字段）（步骤/账本/并行/清单）+ `example/train_tiny.py`（build.py 只读源码，不执行）+ `example/train_sim.py`（build 期运行，注入真实输出）
- 学习闭环：前测/后测分数、所选 GPU、访问过的视图/步骤都存 `localStorage`（key `learnkvc.progress.v1`），
  学习反馈页据此生成前后对照；视图/步骤的访问由 `app.js` 在切换时通过 `LK.progress` 记录
- 状态：`web/tools/collect_traces.py`
  - 装了 PyTorch：跑真实 `generate()` 得到 tokens / logits / 计数
  - 没装：用 README 记录的 token 序列 + `print_summary()` 的断言公式生成解析 trace（文件里会标注 `source: analytic`）
  - 另外生成「多轮 / 多请求」的复用账本（`reuse` 字段），供 Prefix 章第 11 步「多次提问」使用

## 文件

```
web/
├── index.html          # 页面壳（无框架、无 CDN）
├── DESIGN.md           # 设计稿（思考优先的教学逻辑、同步机制、P0–P5）
├── assets/
│   ├── style.css
│   ├── app.js          # hash 路由、五个学习章节 + 工具视图、步骤/滚动联动
│   ├── codeview.js     # Python 高亮 + 行号 + 代码区滚动
│   ├── workshop.js     # 请回答 / 多次提问 / 归因 / 总结 / 作业
│   ├── anatomy.js      # 模型组成 / 层与维度 / 参数→token / GPU 加载 / kernel 表
│   ├── train.js        # 训练章节：账本 / 计算器 / 并行 / 训练与仿真代码
│   ├── net.js          # 推理网络章节：NCCL 模拟 / RoCE / PD 计算器 / 仿真代码
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
    └── anatomy.json
```
