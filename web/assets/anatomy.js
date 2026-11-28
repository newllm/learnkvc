/* ④ Model anatomy: composition, layers vs dimensions, parameters -> token,
 * GPU loading, and the kernels a token generation touches. */
(function () {
  const LK = (window.LK = window.LK || {});
  const A = (LK.anatomy = LK.anatomy || {});
  const fmt = (n) => Number(n).toLocaleString("en-US");
  const esc = (s) =>
    String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const go = (view, extra) =>
    document.dispatchEvent(new CustomEvent("lk:navigate", { detail: Object.assign({ view }, extra || {}) }));

  const state = {
    presetId: "qwen30b",
    layer: 0,
    step: 0,
    mode: "decode",
    ctx: 32768,
    rtCtx: 32768,
    rtBatch: 1,
    quant: "q4",
    quiz: {},
    overrides: {}, // {presetId, layers, kv_heads, ctx} from the sliders
  };

  const calc = {
    params(n) {
      if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
      if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
      if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
      return String(n);
    },
    gb(v) {
      if (v >= 100) return v.toFixed(0) + " GB";
      if (v >= 1) return v.toFixed(1) + " GB";
      return v.toFixed(2) + " GB";
    },
    bytes(kib) {
      if (kib >= 1024 * 1024) return (kib / 1024 / 1024).toFixed(2) + " GiB";
      if (kib >= 1024) return (kib / 1024).toFixed(1) + " MiB";
      return kib.toFixed(0) + " KiB";
    },
    kib(ctx, kb) {
      return (ctx * kb) / 1024;
    },
  };

  function math(c) {
    const h = c.hidden;
    const q_dim = c.q_heads * c.head_dim;
    const kv_dim = c.kv_heads * c.head_dim;
    const attn = h * q_dim + 2 * (h * kv_dim) + q_dim * h + 2 * c.head_dim;
    const router = c.experts ? h * c.experts : 0;
    const ffn = c.experts
      ? c.experts * 3 * h * c.expert_inter
      : 3 * h * c.ffn;
    const per_layer = attn + ffn + router;
    const embed = c.vocab * h;
    const head = c.tie ? 0 : c.vocab * h;
    const pos = c.pos_embed ? c.max_pos * h : 0;
    const total = c.layers * per_layer + embed + head + pos;
    const active_layer = attn + router + (c.experts ? c.top_k * 3 * h * c.expert_inter : ffn);
    const active = c.layers * active_layer + embed + head;
    return {
      q_dim, kv_dim, attn, ffn, router, per_layer, embed, head, pos, total, active,
      kv_bytes: 2 * c.layers * c.kv_heads * c.head_dim * 2,
      q4_gb: (total * 4.85) / 8 / 1e9,
      bf16_gb: (total * 2) / 1e9,
    };
  }

  function cfg() {
    const p = LK_DATA.anatomy.presets.find((x) => x.id === state.presetId) || LK_DATA.anatomy.presets[0];
    const o = state.overrides[state.presetId] || {};
    return { preset: p, c: { ...p, layers: o.layers ?? p.layers, kv_heads: o.kv_heads ?? p.kv_heads } };
  }

  A.render = function (el, data) {
    el.__data = data;
    el.__state = state;
    const { preset, c } = cfg();
    const m = math(c);
    const ctx = state.ctx;
    const weightsGb = state.quant === "q4" ? m.q4_gb : m.bf16_gb;
    const kvGib = (ctx * m.kv_bytes) / 1073741824;
    const h2dSec = weightsGb / 25;

    el.innerHTML =
      head(preset) +
      composition(preset, c, m) +
      packageSection(data.files) +
      layersSection(preset, c, m) +
      pipeline(data.timeline, preset, c) +
      gpuLoad(data.gpuLoad, weightsGb, kvGib, h2dSec, state.quant, ctx) +
      kernelsTable(data.kernels) +
      gpuFunctionMap(data.gpuFunctions) +
      runtimeSection(data.runtime, preset, c, m) +
      simSection(data) +
      checkSection(data.notes);

    bind(el, data);
  };

  /* ------------------------------------------------------------- header */

  function head(preset) {
    return (
      '<p class="kicker">LLM · 参数 · 维度 · GPU</p>' +
      '<h1 class="page-title">一个 LLM，一次 token 生成</h1>' +
      '<p class="page-lead">LLM 是一叠「看上下文 + 逐 token 变换」的层；参数是这些层里的数字；' +
      "维度决定每层张量多宽；层数决定重复多少遍。GPU 负责把这些数字变成 kernel 调用，" +
      "引擎决定缓存和调度。先用左边的预设认识规模，再往下看每一步。</p>" +
      '<div class="preset-row">' +
      LK_DATA.anatomy.presets
        .map(
          (p) =>
            '<button class="preset ' + (p.id === preset.id ? "on" : "") +
            '" data-preset="' + p.id + '">' + esc(p.name) + "</button>"
        )
        .join("") +
      "</div>" +
      '<p class="preset-desc">' + esc(preset.desc) + "</p>"
    );
  }

  /* --------------------------------------------------------------- quiz */

  function quiz(data) {
    const cards = data
      .map((q, i) => {
        const answer = state.quiz[q.id];
        const answered = answer !== undefined;
        return (
          '<article class="qcard' + (answered ? " answered" : "") + '">' +
          '<div class="qhead"><span class="qtag">' + esc(q.tag) + "</span>" +
          '<span class="qnum">Q' + (i + 1) + "</span></div>" +
          '<p class="qtext">' + esc(q.q) + "</p>" +
          '<div class="qopts">' +
          q.options
            .map((opt, k) => {
              let cls = "qopt";
              if (answered) {
                if (k === q.answer) cls += " correct";
                if (k === answer && answer !== q.answer) cls += " wrong";
                if (k === answer) cls += " chosen";
              }
              return (
                '<button class="' + cls + '" data-q="' + q.id + '" data-i="' + k + '"' +
                (answered ? " disabled" : "") + ">" +
                String.fromCharCode(65 + k) + ". " + esc(opt) + "</button>"
              );
            })
            .join("") +
          "</div>" +
          (answered ? '<div class="qwhy"><b>参考</b> ' + esc(q.why) + "</div>" : "") +
          "</article>"
        );
      })
      .join("");
    return (
      '<div class="qprogress"><span>请回答 4 个问题，再往下看</span><span class="q-hint">' +
      Object.keys(state.quiz).length + " / " + data.length + " 已回答</span></div>" +
      '<div class="quiz">' + cards + "</div>"
    );
  }

  /* -------------------------------------------------------- composition */

  function composition(preset, c, m) {
    const perLayer = m.per_layer;
    return (
      '<h2 class="sec-title">1 · LLM 由什么组成</h2>' +
      '<p class="page-lead">自回归 LM 的计算图只有四块：<b>embedding</b>（token id → 向量）、' +
      "<b>N 层 DecoderLayer</b>（每层是 attention + FFN/MoE）、<b>final norm</b>、<b>lm_head</b>（hidden → 词表）。" +
      "但「LLM」不等于这张图：参数只是其中一层，外面还有结构、tokenizer/模板/生成配置，以及推理时的运行时状态。</p>" +
      composeGrid(c, m) +
      '<div class="stack">' +
      block("embedding", "[vocab, hidden] → [n, hidden]", calc.params(m.embed) + " 参数") +
      block("× " + c.layers + " DecoderLayer", "每层：attention + " + (c.experts ? "MoE" : "FFN") +
        " → 输出仍 [n, hidden]", calc.params(perLayer) + " 参数/层") +
      block("final norm", "RMSNorm [n, hidden]", "~" + calc.params(c.hidden) + " 参数") +
      block("lm_head", "[hidden, vocab]" + (c.tie ? "（与 embedding 共享）" : ""),
        calc.params(m.head) + " 参数") +
      "</div>" +
      '<div class="params">' +
      paramRow("每层参数", calc.params(perLayer),
        "attention " + calc.params(m.attn) + " + " + (c.experts ? "MoE " : "FFN ") + calc.params(m.ffn) +
        (m.router ? " + router " + calc.params(m.router) : "")) +
      paramRow("总参数", calc.params(m.total),
        c.layers + " × " + calc.params(perLayer) + " + embedding/head" + (m.pos ? " + 位置表" : "")) +
      paramRow("每 token 激活参数", calc.params(m.active),
        c.experts ? "只读 top-" + c.top_k + " 的 expert；dense LLM 这里等于总参数" : "dense：每步读全部") +
      paramRow("KV / token", calc.bytes(m.kv_bytes / 1024),
        "2（K/V）× " + c.layers + " 层 × " + c.kv_heads + " KV 头 × " + c.head_dim) +
      paramRow("权重体积", calc.gb(m.q4_gb) + "（Q4） / " + calc.gb(m.bf16_gb) + "（BF16）",
        "参数量 × 每参数字节；Q4 ≈ 4.85 bit") +
      "</div>" +
      '<p class="note">' + esc(preset.desc) + "</p>"
    );
  }

  function block(title, shape, params) {
    return (
      '<div class="stackblock"><div><b>' + esc(title) + "</b><span>" + esc(shape) + "</span></div>" +
      "<em>" + esc(params) + "</em></div>"
    );
  }

  function paramRow(k, v, sub) {
    return (
      '<div class="prow"><div class="pk">' + esc(k) + '</div><div class="pv">' + esc(v) +
      '</div><div class="ps">' + esc(sub) + "</div></div>"
    );
  }


  /* --------------------------------------------- model package + runtime */

  function composeGrid(c, m) {
    const cards = [
      {
        title: "参数（权重）",
        big: calc.params(m.total) + " 个数字",
        body: "embedding / attention / MoE / norm / lm_head 里的数值；决定「知道什么」。",
        tone: "w",
      },
      {
        title: "结构（超参）",
        big: c.layers + " 层 · hidden " + c.hidden,
        body: c.q_heads + " Q 头 / " + c.kv_heads + " KV 头" +
          (c.experts ? " · " + c.experts + " 专家 top-" + c.top_k : " · FFN " + c.ffn) +
          " · vocab " + fmt(c.vocab) + "；决定参数形状与计算图，本身不占权重显存。",
        tone: "e",
      },
      {
        title: "词表 / 模板 / 配置",
        big: "tokenizer + template + generation",
        body: "决定文本怎么变成 token、system/user 怎么拼、temperature/top_p 怎么采样；是 LLM 包的一部分，但不在计算图里。",
        tone: "e",
      },
      {
        title: "运行时状态",
        big: "KV " + calc.bytes(m.kv_bytes / 1024) + " / token",
        body: "KV cache、激活、workspace、block table、采样 RNG；LLM 文件里没有，推理时必须存在。",
        tone: "k",
      },
    ];
    return (
      '<div class="compose-grid">' +
      cards
        .map(
          (x) =>
            '<div class="compose-card ' + x.tone + '"><h4>' + esc(x.title) + "</h4>" +
            '<div class="big">' + esc(x.big) + "</div><p>" + esc(x.body) + "</p></div>"
        )
        .join("") +
      "</div>"
    );
  }

  function packageSection(files) {
    return (
      '<h2 class="sec-title">2 · LLM 包里除了权重还有什么</h2>' +
      '<p class="page-lead">下载一个 LLM ≠ 只下载权重。config 决定形状，tokenizer/template 决定输入，' +
      "generation_config 决定采样；同一个权重换模板，行为就变了。</p>" +
      '<div class="table-wrap"><table class="tagtable"><thead><tr>' +
      "<th>文件</th><th>内容</th><th>是参数？</th><th>归属</th></tr></thead><tbody>" +
      files
        .map(
          (f) =>
            "<tr><td><code>" + esc(f.name) + "</code></td><td>" + esc(f.contains) +
            '</td><td class="' + (f.param ? "good" : "") + '">' + (f.param ? "是" : "否") +
            "</td><td>" + esc(f.owner) + "</td></tr>"
        )
        .join("") +
      "</tbody></table></div>" +
      '<p class="note">这些文件加起来是「发布物」；下面第 7 节列的是「运行时现场」，两者不要混。</p>'
    );
  }

  function runtimeSection(rt, preset, c, m) {
    const ctx = state.rtCtx;
    const batch = state.rtBatch;
    const gib = 1024 * 1024 * 1024;
    const kv = (ctx * m.kv_bytes * batch) / gib;
    const chunk = Math.min(ctx, 2048);
    const perTok =
      2 * (c.hidden + m.q_dim + 2 * m.kv_dim +
        (c.experts ? c.top_k * c.expert_inter * 2 : c.ffn));
    const act = (chunk * perTok) / gib;
    const fixed = 1.5;
    const weights = state.quant === "q4" ? m.q4_gb : m.bf16_gb;
    const total = weights + kv + act + fixed;
    const bar = (label, gb, cls) =>
      '<div class="memrow ' + cls + '"><span>' + label + "</span>" +
      '<div class="membar"><i style="width:' + Math.max(1, (gb / total) * 100) + '%"></i></div>' +
      "<b>" + gb.toFixed(2) + " GiB</b></div>";

    return (
      '<h2 class="sec-title">8 · 运行时状态：LLM 文件之外的那一半</h2>' +
      '<p class="page-lead">' + esc(rt.intro) + "它不是参数，也不是权重文件；是「这次请求正在算什么、算到哪了」的现场。" +
      "按活多久分成四组：</p>" +
      '<div class="rt-groups">' +
      rt.groups
        .map(
          (g) =>
            '<div class="rt-card"><b>' + esc(g.label) + "</b><div class=\"rt-chips\">" +
            g.items.map((i) => "<span>" + esc(i) + "</span>").join("") +
            "</div></div>"
        )
        .join("") +
      "</div>" +
      '<h4 class="sec-sub">选上下文和并发，看运行时怎么长</h4>' +
      '<div class="runtime-controls">' +
      '<label class="slider"><span>上下文</span><input type="range" data-rt-ctx min="2048" max="262144" step="2048" value="' + ctx + '"><b>' + fmt(ctx) + '</b></label>' +
      '<label class="slider"><span>并发</span><input type="range" data-rt-batch min="1" max="8" step="1" value="' + batch + '"><b>' + batch + '</b></label>' +
      "</div>" +
      '<div class="membars">' +
      bar("权重", weights, "w") +
      bar("KV cache", kv, "k") +
      bar("激活", act, "a") +
      bar("杂项", fixed, "r") +
      "</div>" +
      '<div class="calcgrid">' +
      stat("权重（固定）", weights.toFixed(1) + " GiB", state.quant === "q4" ? "Q4" : "BF16") +
      stat("KV（ctx × batch）", kv.toFixed(2) + " GiB", fmt(ctx) + " × " + batch + " × " + calc.bytes(m.kv_bytes / 1024) + "/token") +
      stat("prefill 激活", act.toFixed(2) + " GiB", "chunk " + fmt(chunk) + " token × " + calc.bytes(perTok / 1024) + "/token") +
      stat("合计（估算）", total.toFixed(1) + " GiB", "未计框架/分配器碎片") +
      "</div>" +
      '<p class="note">decode 每步只处理 1 个 token，激活远小于 prefill；长上下文时 KV 是最大项（decode 每步还要把整段 KV 读一遍，带宽 bound）。prefix cache 命中时，共享 KV 只存一份。</p>' +
      '<div class="table-wrap"><table class="tagtable"><thead><tr>' +
      "<th>阶段</th><th>分配</th><th>释放</th></tr></thead><tbody>" +
      rt.lifecycle
        .map(
          (x) =>
            "<tr><td><b>" + esc(x.phase) + "</b></td><td>" + esc(x.alloc) +
            "</td><td>" + esc(x.free) + "</td></tr>"
        )
        .join("") +
      "</tbody></table></div>" +
      '<h4 class="sec-sub">三层分工</h4>' +
      '<div class="compose-grid three">' +
      '<div class="compose-card"><h4>LLM 定义</h4><p>层数、hidden、KV 头、词表、dtype：决定「需要哪些状态、长什么样」。</p></div>' +
      '<div class="compose-card"><h4>GPU 提供</h4><p>HBM 存 KV/激活/workspace；kernel 生产与消费这些状态；带宽决定 decode。</p></div>' +
      '<div class="compose-card"><h4>引擎管理</h4><p>block pool / block table / prefix cache / 调度 / 释放 / swap；决定状态放哪、何时复用。</p></div>' +
      "</div>" +
      '<p class="note">' + esc(rt.notes) + "</p>" +
      '<div class="reuse-actions"><button class="btn" id="runtime-calc">去显存计算器，按这张卡算 →</button></div>'
    );
  }

  /* ------------------------------------------------- layers vs dimensions */

  function layersSection(preset, c, m) {
    const L = c.layers;
    state.layer = Math.min(state.layer, L - 1);
    const layerButtons = Array.from({ length: L }, (_, i) =>
      '<button class="layercell' + (i === state.layer ? " on" : "") + '" data-layer="' + i + '">' + i + "</button>"
    ).join("");
    const i = Math.min(state.layer, L - 1);
    const kvPerTokLayer = 2 * c.kv_heads * c.head_dim * 2;

    return (
      '<h2 class="sec-title">3 · 为什么是多层？维度和层是什么关系</h2>' +
      '<p class="page-lead">' + esc(LK_DATA.anatomy.notes.layers) + "</p>" +
      '<div class="layer-tools">' +
      slider("层数", "layers", c.layers, 1, 64, 1) +
      slider("KV 头数", "kv_heads", c.kv_heads, 1, 8, 1) +
      "</div>" +
      '<div class="calcgrid">' +
      stat("总参数", calc.params(m.total), "层数 × 每层 + 词表") +
      stat("KV / token", calc.bytes(m.kv_bytes / 1024), "层数 × KV 头 × head_dim") +
      stat("每层参数", calc.params(m.per_layer), "attention + FFN/MoE") +
      stat("权重 Q4", calc.gb(m.q4_gb), "BF16 " + calc.gb(m.bf16_gb)) +
      "</div>" +
      '<div class="two-col">' +
      '<div class="layergrid">' + layerButtons + "</div>" +
      '<div class="layerinfo">' +
      "<h4>第 " + i + " 层（和第 0 层同形）</h4>" +
      row("输入", "[n, " + c.hidden + "]（残差流）") +
      row("Q", "[n, " + m.q_dim + "] = " + c.q_heads + " 头 × " + c.head_dim) +
      row("K / V", "[n, " + m.kv_dim + "] = " + c.kv_heads + " 头 × " + c.head_dim +
        "（" + (c.q_heads % c.kv_heads === 0
          ? "每 " + (c.q_heads / c.kv_heads) + " 个 Q 头共享 1 个 KV 头"
          : c.q_heads + " 个 Q 头 / " + c.kv_heads + " 个 KV 头") + "）") +
      row("该层 KV cache", "(1, " + c.kv_heads + ", T, " + c.head_dim + ") · " +
        calc.bytes(kvPerTokLayer / 1024) + "/token") +
      row(c.experts ? "MoE" : "FFN", c.experts
        ? c.experts + " 个 expert，每 token 选 " + c.top_k + " 个（每个 " + c.expert_inter + " 宽）"
        : "hidden → " + c.ffn + " → hidden") +
      row("输出", "[n, " + c.hidden + "]（加回残差）") +
      '<p class="note">形状不变，变的是权重数值和这一层能看到的上下文——所以「加深」是多次加工，不是改变维度。</p>' +
      "</div></div>"
    );
  }

  function slider(label, key, value, min, max, step) {
    return (
      '<label class="slider"><span>' + esc(label) + "</span>" +
      '<input type="range" data-tweak="' + key + '" min="' + min + '" max="' + max +
      '" step="' + step + '" value="' + value + '">' +
      "<b>" + value + "</b></label>"
    );
  }

  function stat(label, value, sub) {
    return (
      '<div class="stat"><div class="stat-label">' + esc(label) + "</div>" +
      '<div class="stat-value">' + esc(value) + "</div>" +
      '<div class="stat-sub">' + esc(sub) + "</div></div>"
    );
  }

  function row(k, v) {
    return '<div class="lrow"><span>' + esc(k) + "</span><code>" + esc(v) + "</code></div>";
  }

  /* ------------------------------------------------------------- pipeline */

  function pipeline(timeline, preset, c) {
    const n = state.mode === "decode" ? 1 : 1024;
    const step = timeline[Math.min(state.step, timeline.length - 1)];
    const fill = (s) =>
      String(s)
        .replace(/\{hidden\}/g, c.hidden)
        .replace(/\{q_dim\}/g, c.q_heads * c.head_dim)
        .replace(/\{kv_dim\}/g, c.kv_heads * c.head_dim)
        .replace(/\{kv_heads\}/g, c.kv_heads)
        .replace(/\{head_dim\}/g, c.head_dim)
        .replace(/\{vocab\}/g, c.vocab)
        .replace(/\{experts\}/g, c.experts)
        .replace(/\{expert_inter\}/g, c.expert_inter)
        .replace(/\{top_k\}/g, c.top_k)
        .replace(/\[1, /g, "[" + n + ", ")
        .replace(/ 1 × /g, " " + n + " × ");
    const dots = timeline
      .map((_, i) => '<button class="pipdot' + (i === state.step ? " on" : "") + '" data-step="' + i + '"></button>')
      .join("");
    return (
      '<h2 class="sec-title">4 · 参数如何参与一次 token 生成</h2>' +
      '<p class="page-lead">同一个 forward，用预填充（一次看 n 个 token）或解码（每次 1 个）跑出来。' +
      "参数就是每步被乘的那些矩阵；GPU 负责把每个操作变成 kernel。</p>" +
      '<div class="pipe-controls">' +
      '<div class="mode-tabs"><button data-mode="decode" class="' + (state.mode === "decode" ? "on" : "") + '">decode（1 个 token）</button>' +
      '<button data-mode="prefill" class="' + (state.mode === "prefill" ? "on" : "") + '">prefill（n=1024）</button></div>' +
      '<div class="pipe-nav"><button class="btn" data-pipe="-1">←</button>' +
      "<span>" + (state.step + 1) + " / " + timeline.length + "</span>" +
      '<button class="btn" data-pipe="1">→</button></div>' +
      "</div>" +
      '<div class="pipe-card">' +
      '<div class="pipe-head"><b>' + esc(step.title) + "</b>" +
      '<span class="worker">' + esc(step.worker) + "</span></div>" +
      '<div class="pipe-op">' + esc(step.op) + "</div>" +
      '<div class="pipe-meta"><span>形状 ' + esc(fill(step.shape)) + "</span>" +
      "<span>kernel " + esc(step.kernel) + "</span></div>" +
      '<p class="note">' + esc(step.detail) + "</p>" +
      "</div>" +
      '<div class="pipedots">' + dots + "</div>"
    );
  }

  /* ----------------------------------------------------------- GPU loading */

  function gpuLoad(stages, weightsGb, kvGib, h2dSec, quant, ctx) {
    const sizes = {
      disk: calc.gb(weightsGb),
      ram: calc.gb(weightsGb),
      h2d: calc.gb(weightsGb) + " · ~" + h2dSec.toFixed(1) + " s @25GB/s",
      vram: calc.gb(weightsGb) + " 权重 + " + kvGib.toFixed(2) + " GiB KV + 激活",
    };
    return (
      '<h2 class="sec-title">5 · GPU 如何加载参数</h2>' +
      '<p class="page-lead">磁盘上的文件不是一个整体，而是按张量切好的权重。' +
      "加载 = 读进 CPU 内存 → H2D 拷贝进显存；之后推理只读，不再写回。</p>" +
      '<div class="quant-toggle"><span>上下文：</span>' +
      '<input type="range" class="ctxrange" data-ctx min="2048" max="262144" step="2048" value="' + ctx + '">' +
      "<span>" + fmt(ctx) + " tok</span></div>" +
      '<div class="quant-toggle"><span>权重格式：</span>' +
      '<button data-quant="q4" class="' + (quant === "q4" ? "on" : "") + '">Q4（~4.85 bit）</button>' +
      '<button data-quant="bf16" class="' + (quant === "bf16" ? "on" : "") + '">BF16（16 bit）</button>' +
      "</div>" +
      '<div class="loadflow">' +
      stages
        .map(
          (s) =>
            '<div class="loadcard"><div class="loadnum">' + esc(s.title) + "</div>" +
            '<div class="loadsize">' + esc(sizes[s.sizeKey] || "") + "</div>" +
            '<p class="note">' + esc(s.detail) + "</p></div>"
        )
        .join("") +
      "</div>" +
      '<p class="note">量化版本：llama.cpp/GGUF 在 matmul 里边读边反量化，显存省但 kernel 更忙；' +
      "HF + bitsandbytes 往往先反量化成 BF16，显存会翻倍。vLLM 的 AWQ 则保留 int4 + scales。</p>"
    );
  }

  /* -------------------------------------------------------------- kernels */

  function kernelsTable(rows) {
    return (
      '<h2 class="sec-title">6 · Token 生成调用了哪些 GPU 计算函数</h2>' +
      '<p class="page-lead">同一套流程，prefill 和 decode 的 kernel 一样，瓶颈相反：' +
      "prefill 是大矩阵乘（计算受限），decode 是 GEMV + 读 KV（带宽受限）。</p>" +
      '<div class="table-wrap"><table class="kerneltable"><thead><tr>' +
      "<th>阶段</th><th>操作</th><th>GPU 实现</th><th>prefill</th><th>decode</th><th>谁负责</th>" +
      "</tr></thead><tbody>" +
      rows
        .map(
          (r) =>
            "<tr><td>" + esc(r.stage) + "</td><td>" + esc(r.op) + "</td><td><code>" +
            esc(r.kernel) + "</code></td><td>" + esc(r.prefill) + "</td><td>" +
            esc(r.decode) + "</td><td>" + esc(r.worker) + "</td></tr>"
        )
        .join("") +
      "</tbody></table></div>"
    );
  }

  /* ------------------------------------------------------ GPU function map */

  function gpuFunctionMap(fn) {
    const secNo = 7;
    const groups = fn.groups
      .map(
        (g, i) =>
          '<h3 class="sec-sub fn-sub" data-toc-sub>' + secNo + "." + (i + 1) + " " + esc(g.title) + "</h3>" +
          '<p class="note">' + esc(g.desc) + "</p>" +
          '<div class="table-wrap"><table class="fntable"><thead><tr>' +
          "<th>操作</th><th>PyTorch / 引擎调用</th><th>CUDA / 库函数</th><th>硬件单元</th><th>说明</th>" +
          "</tr></thead><tbody>" +
          g.rows
            .map(
              (r) =>
                "<tr><td>" + esc(r.op) + "</td><td><code>" + esc(r.call) + "</code></td>" +
                "<td><code>" + esc(r.cuda) + "</code></td><td>" + esc(r.unit) + "</td>" +
                "<td>" + esc(r.note) + "</td></tr>"
            )
            .join("") +
          "</tbody></table></div>"
      )
      .join("");

    return (
      '<h2 class="sec-title">' + secNo + ' · GPU 函数地图：CUDA / kernel / 矩阵计算</h2>' +
      '<p class="page-lead">' + esc(fn.intro) + "</p>" +
      '<div class="fnlayers">' +
      fn.layers
        .map(
          (l, i) =>
            '<div class="fnlayer"><b>' + (i + 1) + " · " + esc(l.name) + "</b>" +
            "<p>" + esc(l.desc) + "</p><span>" + esc(l.examples) + "</span></div>"
        )
        .join("") +
      "</div>" +
      groups +
      '<h3 class="sec-sub fn-sub" data-toc-sub>' + secNo + "." + (fn.groups.length + 1) + " " +
      esc(fn.chain.title) + "</h3>" +
      '<p class="note">' + esc(fn.chain.note) + "</p>" +
      '<ol class="fnchain">' +
      fn.chain.steps
        .map((s) => "<li><b>" + esc(s.op) + "</b> → <code>" + esc(s.call) + "</code></li>")
        .join("") +
      "</ol>" +
      '<div class="fncaveats"><b>口径与常见误区</b><ul>' +
      fn.caveats.map((c) => "<li>" + esc(c) + "</li>").join("") +
      "</ul></div>"
    );
  }

  /* ---------------------------------------------------------------- sim + check */

  function simSection(data) {
    return (
      '<h2 class="sec-title">9 · 仿真代码：参数 / 层数 / KV 账本</h2>' +
      '<p class="page-lead">把四个预设的参数量、激活参数、每 token KV 算成一张表：' +
      "参数决定显存，激活参数决定每步读多少；KV 只随层数、KV 头数和精度线性增长。</p>" +
      '<pre class="train-run"><code>python3 example/structure_sim.py</code></pre>' +
      '<div class="train-code-head"><span>' + esc((data.simCode && data.simCode.file) || "example/structure_sim.py") +
      '</span><span>纯 Python · 可运行</span></div>' +
      '<div class="train-code" id="anatomy-sim-code"></div>' +
      '<h3 class="sec-sub">示例输出</h3><pre class="train-run"><code>' + esc(data.simOutput || "") + "</code></pre>"
    );
  }

  function checkSection(notes) {
    return (
      '<h2 class="sec-title">10 · 自查：把结构、参数和运行时说清楚</h2>' +
      '<ul class="train-rules train-checklist">' +
      '<li>能说出参数、结构超参、运行时状态三者的边界</li>' +
      '<li>能解释层数增加时 hidden 不变，但参数量和 KV 线性增加</li>' +
      '<li>能用 KV/token = 2 × layers × kv_heads × head_dim × 精度 复算 96 KiB</li>' +
      '<li>能解释 MoE 的总参数与激活参数为什么不同</li>' +
      '<li>能说清权重从磁盘到 GPU 显存经过哪些站</li>' +
      '</ul>' +
      details("为什么层数增加，参数量和 KV 都线性增加，但 hidden 不变？", notes.layers + " " + notes.kv) +
      details("参数是什么？它和「知识」是什么关系？", notes.params) +
      details("LLM、GPU、引擎在这套流程里各做了什么？", notes.gpu)
    );
  }

  function details(q, a) {
    return '<details class="thinkbox"><summary>' + esc(q) + "</summary><p>" + esc(a) + "</p></details>";
  }

  /* --------------------------------------------------------------- events */

  function bind(el, data) {
    el.querySelectorAll("[data-preset]").forEach((b) =>
      b.addEventListener("click", () => {
        state.presetId = b.dataset.preset;
        state.layer = 0;
        state.overrides = {};
        A.render(el, data);
      })
    );
    el.querySelectorAll("[data-layer]").forEach((b) =>
      b.addEventListener("click", () => {
        state.layer = +b.dataset.layer;
        A.render(el, data);
      })
    );
    el.querySelectorAll("[data-tweak]").forEach((inp) =>
      inp.addEventListener("change", () => {
        const p = state.overrides[state.presetId] || {};
        p[inp.dataset.tweak] = +inp.value;
        state.overrides[state.presetId] = p;
        A.render(el, data);
      })
    );
    el.querySelectorAll("[data-mode]").forEach((b) =>
      b.addEventListener("click", () => {
        state.mode = b.dataset.mode;
        A.render(el, data);
      })
    );
    el.querySelectorAll("[data-pipe]").forEach((b) =>
      b.addEventListener("click", () => {
        state.step = Math.max(0, Math.min(data.timeline.length - 1, state.step + +b.dataset.pipe));
        A.render(el, data);
      })
    );
    el.querySelectorAll("[data-step]").forEach((b) =>
      b.addEventListener("click", () => {
        state.step = +b.dataset.step;
        A.render(el, data);
      })
    );
    el.querySelectorAll("[data-ctx]").forEach((inp) =>
      inp.addEventListener("change", () => {
        state.ctx = +inp.value;
        A.render(el, data);
      })
    );
    el.querySelectorAll("[data-quant]").forEach((b) =>
      b.addEventListener("click", () => {
        state.quant = b.dataset.quant;
        A.render(el, data);
      })
    );
    el.querySelectorAll("[data-rt-ctx]").forEach((inp) =>
      inp.addEventListener("change", () => {
        state.rtCtx = +inp.value;
        A.render(el, data);
      })
    );
    el.querySelectorAll("[data-rt-batch]").forEach((inp) =>
      inp.addEventListener("change", () => {
        state.rtBatch = +inp.value;
        A.render(el, data);
      })
    );
    const rtCalc = el.querySelector("#runtime-calc");
    if (rtCalc) rtCalc.addEventListener("click", () => go("calc"));
    const simCode = el.querySelector("#anatomy-sim-code");
    if (simCode && LK.codeview && data.simCode) LK.codeview.mount(simCode, data.simCode.lines);
  }
})();
