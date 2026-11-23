# AGENTS.md

这份文件是 `learnkvc` 仓库（尤其是 `web/` 交互式阅读器）的开发约定。改新功能前先读一遍；和 `web/DESIGN.md`、`web/README.md` 冲突时，以本文件的产品/UI 要求 + DESIGN 的机制设计为准，并在 PR 里同步更新它们。

## 1. 仓库是什么

`learnkvc` 用三份可运行的示例讲 KV cache：

- `example/kv_cache.py`：单请求内部的重算 vs 缓存；
- `example/prefix_cache_demo.py`：多请求共享前缀（块池、哈希链、引用计数、LRU）；
- `example/paged_attention.py`：PagedAttention 的 block table 与 gather。

`web/` 是零依赖静态交互页，**教学主线是「思考」而不是「读文档」**：

```
请回答（前测） → 三章四段式（含多次提问、三层归因） → 模型解剖 / 计算器 → 后测 → 学习反馈 → 课后作业
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
| 题目/前测/后测/归因/反思 | `web/tools/pedagogy.json` |
| 模型预设/流水线/GPU 档位/模型包/运行时 | `web/tools/anatomy.json` |
| 运行状态 | `web/tools/collect_traces.py` |

- 生成物 `web/data/data.js`、`web/data/traces.js` **提交进仓库**，保证双击可用；**禁止手改**，改源文件后跑 `build.py`。
- `build.py` 对模型规模有断言（30.5B / 3.35B / 96 KiB、7B 56 KiB、32B 256 KiB）；prefix 由真实 demo 运行、paged 由纯记账生成，没有硬断言。改数字时要同步核对关键值（prefix 命中率 52% / 18% / 33%，paged 9 个快照、seq-c 复用 b2/b3），不能只改展示。
- 代码引用用符号：`function:generate`、`class:PrefixCache.acquire`、`docstring:module`；section 用**标题原文**匹配。
- 示例文件必须保持可独立运行（`python3 example/kv_cache.py`），不要引入只服务于网页的改动。

## 4. 学习闭环（功能要求）

1. **学习设定**：请回答页顶部固定展示锚点模型（Qwen3-Coder-30B-A3B）规格与可选 GPU（显存/带宽/BF16 算力/PCIe），并给出「Q4 + 上下文 KV + 运行时 ≈ 多少、能否跑、decode 上限、prefill 下限」。
2. **前测**：6 题、评分；**提交前不显示对错和解释**；分数与答案存 `localStorage`，作为基线。可重做。
3. **章节四段式（硬性）**：每章必须按「请回答 → 探索阅读 → 反思与归因 → 总结归纳」组织。
   - 请回答：3 道预测题，选完才揭示，不计分；
   - 探索阅读：现有 steps 代码 ↔ 讲解 ↔ 可视化联动；
   - 反思与归因：2–4 个开放问题 + 可折叠参考；再对 3 张卡做「模型 / GPU / 引擎」归因，提交归因后才能完成本阶段；
   - 总结归纳：参考总结（概要 + 详细）+ 用户自己的总结（textarea）+「完成本章」按钮。
   - 章节导航统一在左侧轨道（可收缩）：顶部是「章节」切换（三章标题 + ①②③，当前高亮、完成打勾，点击切章并保留各章阶段），下面是四阶段（请回答/探索阅读/反思与归因/总结归纳），探索阶段下面列出步骤；非探索阶段只显示章节、四阶段与提示，隐藏步骤列表、代码 pane、可视化、底部控制条；
   - **多次提问不单独成页**：作为 Prefix 章「探索阅读」中的一个步骤（内嵌多轮/多请求模拟器）；
   - **三层归因不单独成页**：并入每章「反思与归因」阶段；「总结归纳」只保留参考总结、用户总结与完成本章；跨章综合在课后作业的对照矩阵。
4. **后测**：6 题同概念换问法、评分；提交后生成反馈。
5. **反馈**：前测/后测分数、变化、按概念对照（提升/维持/退步/仍需复习）、学习完成度（步骤/视图）、错题回看按钮；6 个概念覆盖三章 + 模型解剖（参数/结构/模型包/运行时），完成后自测总步数为三章之和（45）。
6. **课后作业**：三章「总结归纳」全部完成后解锁；内容是迁移到其他领域（CPU cache / DB buffer pool / CDN / Git packfile / 浏览器缓存任选），含三章对照矩阵 + 模型解剖对照矩阵（参数/结构/模型包/运行时）、6 道题（迁移 / KV 计算 / TTL 设计 / 自由类比 / 模型包体检 / 运行时账本）、参考答案与提交记录。未完成时显示锁定的章节清单和进度。
7. **问题两层**：每道题必须有 `summary`（一句话「考什么」）和 `detail`（可展开的「背景 / 怎么想」，不泄露答案）。
8. **总结两层**：反馈页总结必须是 `brief`（一句话） + `detail`（详细描述）可折叠；用户先写自己的总结，再点开对照。
9. **方向**：先形成假设 → 用代码/状态验证 → 归因到「模型 / GPU / 引擎」→ 用自己的话总结。禁止把文档内容直接当课文投喂。

### localStorage 与路由

- `learnkvc.progress.v1`：`{ pre, post, gpu, views[], steps[], lessons, homework }`。
  - `lessons.<chapter>`：`{think, thinkAnswers, exploreMax, reflect, reflectText, attribution, attributionSubmitted, summary, summaryText, summaryAt}`；
  - `homework`：`{answers, submitted, at}`；
  - 前/后测分数和答案、所选 GPU、访问记录都在这一个 key 里。
- `learnkvc.theme`：`dark` / `light`。
- `learnkvc.stepnav`：`{kv,prefix,paged}` 的导航折叠状态。
- `learnkvc.sectionnav`：`{anatomy,calc}` 的模型工具小节导航折叠状态。
- 路由：`#think`、`#kv/N`、`#kv/think|explore|reflect|summary`（prefix/paged 同理）、`#anatomy`、`#calc`、`#feedback`、`#homework`；旧链接 `#reuse` 重定向到 Prefix 章的多次提问步骤，`#reflect` 重定向到 KV 章总结。
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
7. 验收：`build.py` 通过、`node --check` 通过、CDP 遍历无 error、`file://` 可开、760px 布局可用；
8. 四段式验收：每章左侧导航有 4 个阶段按钮（请回答/探索阅读/反思与归因/总结归纳）、请回答阶段答完出 CTA、反思可写、总结完成写 `lessons.<chapter>.summary`；三章完成后 `#homework` 解锁（三章矩阵 + 模型解剖矩阵 + 6 题 + 提交）。

## 6. UI / 设计要求

### 视觉与主题

- **暗色默认**；右上角切换亮/暗，写入 `localStorage`。暗色下代码区深底浅字，**亮色下页面/文档/控件/代码区都变亮**（浅底 + 深色语法高亮、浅色行号与高亮行），不要保留「亮色页面 + 深色代码」的混合态。
- 两套主题都必须检查对比度：正文、`strong`、状态色（good/bad/warn）、行内 code、按钮 primary、气泡、块池状态色都不能出现看不清的文字；改主题色/新增组件后至少用截图或对比度扫描验证一次。
- 所有颜色走 CSS 变量（`--bg/--panel/--line/--fg/--accent/...`）；新增颜色优先加变量，不要散落硬编码。
- **滚动条暗色**：`color-scheme: dark` + `scrollbar-color` + `::-webkit-scrollbar`（轨道 `#0b1119`、滑块 `#26384a`）；新增可滚动容器不需要额外写，继承全局规则。
- 语义色固定：prompt / 生成 token、命中块、新分配块、被淘汰块各有固定色；不要只靠颜色表达状态（同时加文字/形状）。
- 字体：正文系统无衬线，代码 `ui-monospace`；代码 12.5px、正文 13–14px。

### 布局与交互

- 顶栏：品牌 + 导航 + 主题开关。导航一级是「请回答 · 前测 / 学习反馈 / 课后作业」；三个章节收进「章节」下拉，模型解剖与显存计算器收进「模型工具」下拉。触发按钮显示当前所在项（如「章节 · KV cache」），`aria-expanded` 同步；点外部、Esc、选中任一项都关闭。窄屏（≤980px）导航换行，不再横向滚动，下拉菜单要夹在视口内。章节视图是「左侧步骤导航（可收缩）+ 中间代码 + 右侧讲解/可视化」；底部只保留上一步/下一步和计数器，步骤点不再放底部。
- 步骤导航：左侧轨道顶部先是「章节」切换（三章标题 + ①/②/③，当前高亮、完成后打勾；点击切章并保留该章上次所处阶段；折叠时只留 ①②③），然后是四阶段（带完成状态与探索进度），探索阶段下面列出全部步骤（标题、进行中/已完成）；折叠按钮把宽度收到 36px 轨道，状态存 `learnkvc.stepnav`；窄屏（≤980px）默认收起，展开后章节为 3 列、阶段为 4 列（隐藏进度文字、保留完成打勾）、步骤列表限高可滚动。非探索阶段仍显示章节与四阶段，只隐藏步骤列表。
- 模型工具（模型解剖 / 显存计算器）沿用章节导航的轨道语言（`assets/sectionnav.js`）：200px 左栏（面板底色 + 右边框 + 「小节导航 / 当前 / 总数 / 折叠按钮」）、编号 01…、GPU 函数地图下 6 个二级小节（7.1–7.6）；滚动高亮、点击跳转、换预设/拉滑杆后自动重建；折叠到 36px 只留编号，状态存 `learnkvc.sectionnav`；≤980px 默认收起成顶部条，展开后列表限高滚动。
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
- 新页面/新脚本保持顺序加载：`data.js → traces.js → codeview.js → viz/*.js → workshop/anatomy/calc → sectionnav.js → app.js`。
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

- 已完成：P0/P1（KV 章、前测/后测/反馈/学习设定）、P2（Prefix 章）、P3（Paged 章）、P4（显存计算器 + 亮色主题 + 响应式 + 无障碍基础）、P5（根 README 入口 + GitHub Pages workflow）、每章四段式、多次提问并入 Prefix 章、反思与归因合并、三章后的迁移课后作业、模型工具左侧小节导航 + GPU 函数地图。
- 已知缺口（可继续做）：
  - 章节级前测/后测目前是全局一套；如需更细，可给每章 2–3 题并复用 `pedagogy.json` 结构；
  - GitHub Pages 需要在仓库 Settings → Pages 选择 “GitHub Actions” 才会真正部署；
  - 各章总结/作业可以进一步做「导出 Markdown」；
  - 移动端代码 pane 可折叠成抽屉（目前固定 42vh）。
- 改动这些内容时，优先补测试、补文档，再动 UI。
