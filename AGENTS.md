# AGENTS.md

这份文件是 `learnkvc` 仓库（尤其是 `web/` 交互式阅读器）的开发约定。改新功能前先读一遍；和 `web/DESIGN.md`、`web/README.md` 冲突时，以本文件的产品/UI 要求 + DESIGN 的机制设计为准，并在 PR 里同步更新它们。

## 1. 仓库是什么

`learnkvc` 用三份 KV cache 示例 + 最小训练脚本 + 十份纯 Python 仿真讲「推理与训练」，网页把认识 GPU（GPU / NPU 与国产算力 / CUDA）、认识 LLM（Transformer 与 LLM / LLM 结构 / LLM 训练）、显存计算器（工具入口）、KV cache、推理网络（总览 / NCCL / ECN+DCQCN）、LLM 优化组织成学习章节：

- `example/kv_cache.py`：单请求内部的重算 vs 缓存；
- `example/prefix_cache_demo.py`：多请求共享前缀（块池、哈希链、引用计数、LRU）；
- `example/paged_attention.py`：PagedAttention 的 block table 与 gather；
- `example/train_tiny.py`：最小 PyTorch 训练循环（forward/loss/backward/optimizer、梯度累积、显存账本）；需要 torch，`build.py` 只读源码不执行。
- `example/net_sim.py`：纯 Python 模拟 NCCL ring 与 PD KV 传输，打印步数/流量/时间/TTFT。
- `example/train_sim.py`：纯 Python 模拟训练显存账本（DP/ZeRO/TP/PP/激活重算），打印每卡峰值。
- `example/nccl_sim.py`：纯 Python 模拟 ring/tree all-reduce、α-β 消息大小扫描、collectives 账本与 TP/decode 通信量。
- `example/cuda_sim.py`：纯 Python 账本模拟 occupancy、内存层级、coalescing、roofline 与 CUDA Graph launch 开销。
- `example/gpu_sim.py`：纯 Python 账本对比 CPU vs GPU（核心/算力/带宽/缓存层级）、LLM roofline、launch/Amdahl 与 CUDA/ROCm/昇腾/HCU 软件栈。
- `example/npu_sim.py`：纯 Python 账本对比 GPU / NPU / 国产加速器的 roofline、decode/prefill 量级与选型。
- `example/transformer_sim.py`：纯 Python 账本：注意力 N²、KV cache、训练 FLOPs 与缩放表。
- `example/structure_sim.py`：纯 Python 账本：LLM 参数 / 激活参数 / KV per token / 层维度。
- `example/ecn_sim.py`：纯 Python 简化仿真 ECN/WRED 标记 + DCQCN 速率收敛 + PFC pause，对照「仅 PFC / ECN+DCQCN / 二者结合」的吞吐、标记率、pause 与丢包。
- `example/opt_sim.py`：纯 Python 优化账本：FlashAttention HBM 流量（naive vs tiled）、量化后 decode 字节/时间、投机解码加速比与其他优化攻击的瓶颈。

`web/` 是零依赖静态交互页，**教学主线是「思考」而不是「读文档」**：

```
请回答（前测） → 认识 GPU（GPU / NPU 与国产算力 / CUDA 执行模型，三章三段式） → 认识 LLM（Transformer 与 LLM / LLM 结构 / LLM 训练） → 显存计算器（工具入口） → 认识 KVCache（三章三段式） → 推理网络（总览 / NCCL 学习章节 / ECN + DCQCN，三章三段式） → LLM 优化（三段式） → 后测 → 学习反馈
```

## 2. 常用命令

```bash
# 1) 重新生成页面数据（改了 example/*.py、*_READING.md、web/tools/*.json 之后必须跑）
python3 web/tools/build.py

# 2) 直接看：双击 web/index.html（file:// 必须可用）
#    或起静态服务
python3 -m http.server 8000
# http://127.0.0.1:8000/web/index.html

# 3) 语法检查（无 package.json、无 npm 依赖）
for f in web/assets/*.js web/assets/viz/*.js; do node --check "$f" || exit 1; done

# 4) 页面回归（headless Chrome + CDP；本机 Chrome 可用时）
#    建议脚本：启动 --remote-debugging-port=9222，Runtime.evaluate 遍历 tabs，
#    断言每个视图 active、控制台 0 error、关键计数/形状正确。
```

环境里没有 PyTorch 时：KV 章的 trace 用 `print_summary` 的断言公式解析生成（标注 `source: analytic`）；Prefix 章直接 import 纯标准库 demo 真实运行；Paged 章用纯记账复刻 `main()`。**不要为了「有 torch」而改示例代码。**

## 3. 单一事实来源（硬性）

| 内容 | 唯一来源 |
| --- | --- |
| 代码展示 | `example/*.py`（AST 解析符号区间，**不要在示例里加 web 标记注释**） |
| 讲解卡 | `example/*_READING.md`（按 `##`/`###` 切节） |
| 步骤映射 | `web/tools/steps.json`、`steps.prefix.json`、`steps.paged.json` |
| 题目/前测/后测/章节测验/归因/反思 | `web/tools/pedagogy.json`；`transformer`/`anatomy`/`net`/`train`/`nccl`/`cuda`/`ecn`/`gpu`/`npu`/`opt` 的 lesson 随各自的 `<chapter>.json`，`build.py` 合并进 `data.lessons` |
| LLM 结构（预设/流水线/GPU 档位/LLM 包/运行时/lesson） | `web/tools/anatomy.json` + `example/structure_sim.py` + `assets/anatomy.js` |
| 训练教程 / 训练计算器 / 章节 lesson | `web/tools/train.json` + `example/train_tiny.py` + `example/train_sim.py` |
| 推理网络总览 / NCCL 模拟 / PD 分离 / 章节 lesson | `web/tools/net.json` + `example/net_sim.py` |
| NCCL 学习章节（collectives / ring tree / α-β / 拓扑 / lesson） | `web/tools/nccl.json` + `example/nccl_sim.py` + `assets/nccl.js` |
| CUDA 执行模型（warp / 访存 / occupancy / roofline / graph / lesson） | `web/tools/cuda.json` + `example/cuda_sim.py` + `assets/cuda.js` |
| 认识 GPU（CPU/GPU/NPU 分工、LLM 适配、显存与 L1/L2 cache、CUDA/ROCm/昇腾/HCU 生态、lesson） | `web/tools/gpu.json` + `example/gpu_sim.py` + `assets/gpu.js` |
| NPU 与国产算力（GPU/NPU/ASIC 执行模型、厂商地图、软件栈、LLM 选型、lesson） | `web/tools/npu.json` + `example/npu_sim.py` + `assets/npu.js` |
| Transformer 与 LLM（自注意力、架构、缩放定律、为什么 LLM 胜出、lesson） | `web/tools/transformer.json` + `example/transformer_sim.py` + `assets/transformer.js` |
| ECN + DCQCN（交换机/服务器配置、组网、无损、优先级/PFC、ECN 阈值、lesson） | `web/tools/ecn.json` + `example/ecn_sim.py` + `assets/ecn.js` |
| LLM 优化（FlashAttention / PagedAttention / 量化 / 投机解码 / 调度与结构优化 / 选型 / lesson） | `web/tools/opt.json` + `example/opt_sim.py` + `assets/opt.js` |
| 运行状态 | `web/tools/collect_traces.py` |

- 生成物 `web/data/data.js`、`web/data/traces.js` **提交进仓库**，保证双击可用；**禁止手改**，改源文件后跑 `build.py`。`build.py` 会运行 `net_sim.py` / `train_sim.py` / `nccl_sim.py` / `cuda_sim.py` / `gpu_sim.py` / `npu_sim.py` / `transformer_sim.py` / `structure_sim.py` / `ecn_sim.py` / `opt_sim.py`，把真实 stdout 注入 `simOutput`。
- `build.py` 对 LLM 规模有断言（30.5B / 3.35B / 96 KiB、7B 56 KiB、32B 256 KiB）；prefix 由真实 demo 运行、paged 由纯记账生成，没有硬断言。改数字时要同步核对关键值（prefix 命中率 52% / 18% / 33%，paged 9 个快照、seq-c 复用 b2/b3），不能只改展示。
- 代码引用用符号：`function:generate`、`class:PrefixCache.acquire`、`docstring:module`；section 用**标题原文**匹配。
- 示例文件必须保持可独立运行（`python3 example/kv_cache.py`），不要引入只服务于网页的改动。

## 4. 学习闭环（功能要求）

1. **学习设定**：请回答页顶部固定展示锚点 LLM（Qwen3-Coder-30B-A3B）规格与可选 GPU（显存/带宽/BF16 算力/PCIe），并给出「Q4 + 上下文 KV + 运行时 ≈ 多少、能否跑、decode 上限、prefill 下限」。下方是**学习路线图**：5 阶段 / 13 章，章节节点直达 `#<chapter>/think`，显存计算器作为工具链接；完成章节显示 ✓。
2. **前测**：10 题、评分；**提交前不显示对错和解释**；分数与答案存 `localStorage`，作为基线。可重做。
3. **章节三段式（硬性）**：十三个学习章节（Transformer 与 LLM / LLM 结构 / GPU / NPU 与国产算力 / CUDA 执行模型 / KV / Prefix / Paged / 推理网络总览 / NCCL 学习章节 / ECN + DCQCN / LLM 训练 / LLM 优化）都必须按「请回答 → 探索阅读 → 反思与总结」组织；`transformer`/`anatomy`/`net`/`train`/`nccl`/`cuda`/`ecn`/`gpu`/`npu`/`opt` 是文档型章节（无代码 pane，探索阶段是整篇教程 + 分节跳转）。
   - 请回答：3 道预测题，选完才揭示，不计分；
   - 探索阅读：现有 steps 代码 ↔ 讲解 ↔ 可视化联动；
   - 反思与总结：2–4 个开放问题 + 可折叠参考；3 张归因卡（LLM / GPU / 引擎）；参考总结（概要 + 详细）+ 用户自己的总结（textarea）+ 2 题章节测验（提交后计分并显示解释）+「导出 Markdown」+「完成本章」按钮；提交归因后才能完成本章，导出内容包含总结、反思、归因、测验和整章教程内容。
   - 章节导航统一在左侧轨道（可收缩）：KV 三章的切换放在顶栏「认识 KVCache」菜单，左侧只保留三阶段（请回答/探索阅读/反思与总结），探索阶段下面列出步骤（KV 是代码步骤，文档章节是教程小节）；非探索阶段仍显示三阶段，隐藏步骤列表、代码 pane、可视化、底部控制条；
   - **多次提问不单独成页**：作为 Prefix 章「探索阅读」中的一个步骤（内嵌多轮/多请求模拟器）；
   - `transformer`/`anatomy`/`net`/`train`/`nccl`/`cuda`/`ecn`/`gpu`/`npu` 的请回答/探索/反思与总结内容在各自主 JSON 的 `lesson` 字段，构建时合并；探索阶段保留 NCCL 模拟器、PD 计算器、训练计算器、NCCL 逐步播放器、CUDA occupancy/coalescing/roofline/graph 互动、ECN/WRED 阈值与 PFC/DCQCN 场景互动、认识 GPU 的 CPU/GPU 对比与存储层级互动，以及 `example/<chapter>_sim.py` 仿真代码与 build 期输出；
   - **三层归因不单独成页**：并入每章「反思与总结」阶段；「总结」包含参考总结、用户总结、章节测验与 Markdown 导出；跨章综合放在学习反馈的「按概念对照」里。
4. **后测**：10 题同概念换问法、评分；提交后生成反馈。
5. **反馈**：前测/后测分数、变化、按概念对照（提升/维持/退步/仍需复习）、学习完成度（步骤/视图）、章节测验得分表、错题回看按钮；10 个概念覆盖认识 GPU、KV 三章、推理网络、认识 LLM 与 LLM 优化，学习视图覆盖 14 个入口；自测总步数为三章 45 + 推理网络 7 + NCCL 9 + CUDA 9 + ECN 10 + 认识 GPU 10 + NPU 8 + Transformer 9 + LLM 结构 10 + LLM 训练 6 + LLM 优化 11 = 134。
6. **问题两层**：每道题必须有 `summary`（一句话「考什么」）和 `detail`（可展开的「背景 / 怎么想」，不泄露答案）。
7. **总结两层**：反馈页总结必须是 `brief`（一句话） + `detail`（详细描述）可折叠；用户先写自己的总结，再点开对照。
8. **方向**：先形成假设 → 用代码/状态验证 → 归因到「LLM / GPU / 引擎」→ 用自己的话总结。禁止把文档内容直接当课文投喂。

### localStorage 与路由

- `learnkvc.progress.v1`：`{ pre, post, gpu, views[], steps[], lessons }`。
  - `lessons.<chapter>`：`{think, thinkAnswers, exploreMax, reflect, reflectText, attribution, attributionSubmitted, summary, summaryText, summaryAt, quizAnswers, quizScore, quizSubmitted}`；
  - 前/后测分数和答案、所选 GPU、访问记录都在这一个 key 里。
- `learnkvc.theme`：`dark` / `light`。
- `learnkvc.stepnav`：`{kv,prefix,paged,transformer,anatomy,net,train,nccl,cuda,ecn,gpu,npu,opt}` 的章节导航折叠状态。
- `learnkvc.sectionnav`：`{calc}` 的工具小节导航折叠状态（`anatomy` 已改为章节）。
- 路由：`#transformer/N`、`#transformer/think|explore|reflect`、`#npu/N`、`#npu/think|explore|reflect`、`#think`、`#kv/N`、`#kv/think|explore|reflect`（旧 `#kv/summary` 兼容到 `#kv/reflect`）（prefix/paged 同理）、`#net/N`、`#net/think|explore|reflect`、`#train/N`、`#train/think|explore|reflect`、`#nccl/N`、`#nccl/think|explore|reflect`、`#cuda/N`、`#cuda/think|explore|reflect`、`#ecn/N`、`#ecn/think|explore|reflect`、`#gpu/N`、`#gpu/think|explore|reflect`、`#npu/N`、`#npu/think|explore|reflect`、`#opt/N`、`#opt/think|explore|reflect`、`#anatomy/N`、`#anatomy/think|explore|reflect`、`#calc`、`#feedback`；旧链接 `#reuse` 重定向到 Prefix 章的多次提问步骤，`#reflect` 重定向到 KV 章反思与总结。
- 所有 `localStorage` 访问必须 try/catch（`file://` 下可能抛错）。

## 5. 新增章节 checklist

1. 新增 `web/tools/steps.<chapter>.json`：`chapter/title/codeFile/docFile` + 有序 steps；并在 `pedagogy.json` 的 `lessons.<chapter>` 补 `think` / `reflect` / `summary`：
   ```json
   { "id": "hash", "title": "块的身份", "section": "最关键的一行：块的身份",
     "code": ["function:block_hash"], "viz": "prefix", "hint": "一句话导语",
     "scenario": "shared" }
   ```
2. 写 trace：在 `collect_traces.py` 中生成可回放的快照（每步状态、统计、来源），并在 `build.py` 的 `traces.js` 里注册。
3. `build.py` 增加 `data["<chapter>"] = build_chapter("steps.<chapter>.json", think_source=None)`。
4. `index.html`：nav tab + `<section class="view" id="view-<chapter>">`（代码 pane / doc / viz），加载对应 `assets/viz/<chapter>.js`。
5. `app.js`：`VIEWS`、元素、`state.<chapter>`、`render<Chapter>Step/Viz/Counters`、hash route、`navStep`、键盘、timeline 显隐。
6. `style.css`：章节专属样式 + `body[data-view="<chapter>"] .timeline { display:flex }`。
7. 验收：`build.py` 通过、`node --check` 通过、CDP 遍历无 error、`file://` 可开、760px 布局可用；文档章节还要检查 `#<chapter>/N`、`#<chapter>/<stage>` 深链，以及离开章节时对应渲染器 `stop()` 不留定时器；
8. 三段式验收：每章左侧导航有 3 个阶段按钮（请回答/探索阅读/反思与总结）、请回答阶段答完出 CTA、反思可写、完成本章写 `lessons.<chapter>.summary`；每章 2 题章节测验可提交/重做并写入 `quizScore`；「导出 Markdown」能下载含本章内容的 `.md`；`transformer`/`net`/`train`/`nccl`/`cuda`/`ecn`/`gpu`/`npu`/`opt` 也要能完成三阶段。

**文档型章节（transformer / anatomy / net / train / nccl / cuda / ecn / gpu / npu / opt）不走上面的 steps/viz 流程**：没有 `steps.<chapter>.json`、没有 trace、没有代码 pane；探索步骤由教程的 `h2.sec-title` 派生，lesson（think/reflect/attribution/summary）写在对应 `<chapter>.json` 的 `lesson` 字段；`app.js` 用 `DOC_CHAPTERS` + `DOC_SECTIONS` + `DOC_MODULES` + `renderDocStage`，rail 用 `.doc-chapter` / `.doc-main`。新增文档章节要同步 `DOC_CHAPTERS` / `DOC_SECTIONS` / `DOC_MODULES`、`docBody/docMainEl` 的 el、`VIEWS`、`index.html` 视图与 `?v=`，以及 `build.py` 的 `build_doc_sim()`。

## 6. UI / 设计要求

### 视觉与主题

- **暗色默认**；右上角切换亮/暗，写入 `localStorage`。暗色下代码区深底浅字，**亮色下页面/文档/控件/代码区都变亮**（浅底 + 深色语法高亮、浅色行号与高亮行），不要保留「亮色页面 + 深色代码」的混合态。
- 两套主题都必须检查对比度：正文、`strong`、状态色（good/bad/warn）、行内 code、按钮 primary、气泡、块池状态色都不能出现看不清的文字；改主题色/新增组件后至少用截图或对比度扫描验证一次。
- 所有颜色走 CSS 变量（`--bg/--panel/--line/--fg/--accent/...`）；新增颜色优先加变量，不要散落硬编码。
- **滚动条暗色**：`color-scheme: dark` + `scrollbar-color` + `::-webkit-scrollbar`（轨道 `#0b1119`、滑块 `#26384a`）；新增可滚动容器不需要额外写，继承全局规则。
- 语义色固定：prompt / 生成 token、命中块、新分配块、被淘汰块各有固定色；不要只靠颜色表达状态（同时加文字/形状）。
- 字体：正文系统无衬线，代码 `ui-monospace`；代码 12.5px、正文 13–14px。

### 布局与交互

- 顶栏：品牌 + 导航 + 主题开关。五个学习章节下拉 + 一级入口：「认识 GPU」（GPU / NPU 与国产算力 / CUDA 执行模型）、「认识 LLM」（Transformer 与 LLM / LLM 结构 / LLM 训练）、「认识 KVCache」（单请求 KV / Prefix / Paged）、「推理网络」（总览 / NCCL 学习章节 / ECN + DCQCN）、「LLM 优化」（总览 / FlashAttention / LLM 量化 / 投机解码）；一级入口「显存计算器」；一级还有「请回答 · 前测 / 学习反馈」。触发按钮显示当前所在项（如「认识 KVCache · KV」「LLM 训练 · 训练教程」），`aria-expanded` 同步；点外部、Esc、选中任一项都关闭。窄屏（≤980px）导航换行，不再横向滚动，下拉菜单要夹在视口内；KV 代码 pane 在 ≤980px 默认折叠成「展开代码」抽屉，点 header 切换。KV 章节视图是「左侧步骤导航（可收缩）+ 中间代码 + 右侧讲解/可视化」；`transformer`/`anatomy`/`net`/`train`/`nccl`/`cuda`/`ecn`/`gpu`/`npu`/`opt` 是「左侧步骤导航（可收缩）+ 右侧整篇教程」，没有代码 pane；底部只保留上一步/下一步和计数器，步骤点不再放底部。
- 步骤导航：左侧轨道直接是三阶段（带完成状态与探索进度），探索阶段下面列出全部步骤（标题、进行中/已完成）；折叠按钮把宽度收到 36px 轨道，状态存 `learnkvc.stepnav`；窄屏（≤980px）默认收起，展开后阶段为 3 列（隐藏进度文字、保留完成打勾）、步骤列表限高可滚动。非探索阶段仍显示三阶段，只隐藏步骤列表。KV 三章切换用顶栏「认识 KVCache」菜单。
- `transformer`/`anatomy`/`net`/`train`/`nccl`/`cuda`/`ecn`/`gpu`/`npu`/`opt` 已是章节，使用章节左栏：三阶段 + 教程小节列表（每章 6–11 节）；探索阶段点击小节滚动到对应 `h2`，滚动时同步高亮/计数；折叠状态与 KV 章共用一个 `learnkvc.stepnav`。
- 显存计算器仍是工具，沿用章节导航的轨道语言（`assets/sectionnav.js`）：200px 左栏（面板底色 + 右边框 + 「小节导航 / 当前 / 总数 / 折叠按钮」）、编号 01…；滚动高亮、点击跳转、换预设/拉滑杆后自动重建；折叠到 36px 只留编号，状态存 `learnkvc.sectionnav`；≤980px 默认收起成顶部条，展开后列表限高滚动。
- 交互优先用「步进 / 滑杆 / 按钮」，不要自动播放惊吓用户；播放到末尾要停止并可重播。
- 步骤切换要同步：左侧导航高亮 + 代码高亮/自动滚动 + 讲解卡 + 可视化 + 底部计数器 + URL hash。
- 图表/动画用原生 DOM/SVG/CSS；不做无意义的持续动画。
- 移动端断点 980px：章节上下堆叠、三栏/多列变单列、导航可横滑、表格可横滚。

### 无障碍

- 保留 skip link、`aria-live` 计数器、`aria-pressed`（主题）、导航折叠按钮的 `aria-expanded`/`aria-label`。
- 顶栏下拉是菜单：触发按钮带 `aria-haspopup="menu"` / `aria-expanded`，项为 `role="menuitem"`；↑/↓ 循环、Home/End、Esc 关闭并回到触发按钮、Tab 关闭，点击外部关闭。
- 键盘可用：←/→ 换步骤，空格播放/暂停；`focus-visible` 必须有明显焦点环。
- 尊重 `prefers-reduced-motion`；对比度保持可读；不要移除原生控件的可操作性。

### 技术约束（硬性）

- **零依赖**：不引 React/Vue、npm 包、CDN、外部字体；只用原生 HTML/CSS/JS。
- **经典 `<script>`**（不是 ES module），保证 `file://` 直接打开可用；数据用 `window.LK_DATA` / `window.LK_TRACES` 注入，不用 `fetch`。
- 新页面/新脚本保持顺序加载：`data.js → traces.js → codeview.js → viz/*.js → workshop/anatomy/calc/train/net/nccl/cuda/ecn/gpu/opt → sectionnav.js → app.js`。
- **静态资源带版本号**：`web/index.html` 里 CSS/data/JS 的 URL 都带 `?v=YYYY-MM-DD`。改 `web/assets/**` 或 `web/data/**` 后必须同步更新这个版本号，否则 GitHub Pages 缓存会让新旧 HTML/JS 混用（例如旧 app.js 把无 `data-view` 的下拉触发器当 tab，点出 `#undefined`）。
- 文案中文、代码标识英文；新加的大段说明写进 `web/DESIGN.md` 或 `web/README.md`，不要只藏在代码注释里。

## 7. 测试与验收（Definition of Done）

每次改动至少完成：

1. `python3 web/tools/build.py` 无报错，数据断言通过；
2. `node --check` 覆盖所有改动过的 JS；
3. headless Chrome/CDP 遍历所有 tabs：每个视图 `active`、标题正确、**控制台 0 error**；
4. `file://` 打开 `web/index.html` 能渲染（数据已内联）；
5. 760px 宽度下无横向溢出、章节/计算器变单列；
6. 涉及题目/总结的改动：确认「概要 + 详细描述」都在，前测提交前不泄露答案；
7. 涉及增删文档标题：同步根 `README.md` 的目录锚点与 `web/` 文档。

**不要**在未验证 `file://` 和移动端的情况下宣称完成。

## 8. 维护本文件

改动了 web 的目录结构、命令、数据管线、学习闭环或 UI 约定时，**同步更新 AGENTS.md**；它是后续 agent 的入口，不要让它和实际代码脱节。

## 9. 当前状态与路线

- 已完成：P0/P1（KV 章、前测/后测/反馈/学习设定）、P2（Prefix 章）、P3（Paged 章）、P4（显存计算器 + 亮色主题 + 响应式 + 无障碍基础）、P5（根 README 入口 + GitHub Pages workflow）、每章三段式、多次提问并入 Prefix 章、反思与总结合并、GPU 函数地图、Transformer 与 LLM、LLM 训练与推理网络的三段式章节（NCCL ring 模拟 / RoCE / PD 分离计算器 / 训练计算器 / `net_sim.py` / `train_sim.py` 仿真代码）；推理网络深化两章——NCCL 学习章节（collectives / ring-tree 播放器 / α-β 扫描 / 拓扑 / `nccl_sim.py`，9 节）与 ECN + DCQCN（交换机/服务器配置、leaf-spine/rail-optimized 组网、PFC/ECN 无损、优先级与阈值 / `ecn_sim.py`，10 节）；「认识 GPU」下拉三章——GPU（CPU/GPU/NPU 分工、LLM 适配、显存与 L1/L2 cache、CUDA/ROCm/昇腾/HCU 生态 / `gpu_sim.py`，10 节）、NPU 与国产算力（GPU/NPU/ASIC、厂商地图、软件栈、选型 / `npu_sim.py`，8 节）与 CUDA 执行模型（occupancy / coalescing / roofline / CUDA Graph / `cuda_sim.py`，9 节）；LLM 优化（FlashAttention / PagedAttention / 量化 / 投机解码 / 调度与结构优化 / 选型 / `opt_sim.py`，11 节）；Transformer 与 LLM、LLM 训练归入「认识 LLM」菜单；LLM 结构归入「认识 LLM」三段式章节，显存计算器作为顶栏一级入口；每章 2 题章节测验 + 总结页 Markdown 导出 + 移动端代码抽屉。
- 已知缺口（可继续做）：
  - 全局前测/后测仍是跨章 10 题；每章已有 2 题章节测验，如需更细可扩到 3 题/章；
  - GitHub Pages 需要在仓库 Settings → Pages 选择 “GitHub Actions” 才会真正部署。
- 改动这些内容时，优先补测试、补文档，再动 UI。
