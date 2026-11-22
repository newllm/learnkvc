/* ④ Model anatomy: composition, layers vs dimensions, parameters -> token,
 * GPU loading, and the kernels a token generation touches. */
(function () {
  const LK = (window.LK = window.LK || {});
  const A = (LK.anatomy = LK.anatomy || {});
  const fmt = (n) => Number(n).toLocaleString("en-US");
  const esc = (s) =>
    String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const go = (view) => document.dispatchEvent(new CustomEvent("lk:navigate", { detail: { view } }));

  const state = {
    presetId: "qwen30b",
    layer: 0,
    step: 0,
    mode: "decode",
    ctx: 32768,
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
      quiz(data.quiz) +
      composition(preset, c, m) +
      layersSection(preset, c, m) +
      pipeline(data.timeline, preset, c) +
      gpuLoad(data.gpuLoad, weightsGb, kvGib, h2dSec, state.quant, ctx) +
      kernelsTable(data.kernels) +
      thinkBoxes(data.notes) +
      '<div class="reuse-actions"><button class="btn primary" id="anatomy-reflect">把「模型/GPU/引擎」写进我的总结 →</button></div>';

    bind(el, data);
  };

  /* ------------------------------------------------------------- header */

  function head(preset) {
    return (
      '<p class="kicker">模型 · 参数 · 维度 · GPU</p>' +
      '<h1 class="page-title">一个模型，一次 token 生成</h1>' +
      '<p class="page-lead">模型是一叠「看上下文 + 逐 token 变换」的层；参数是这些层里的数字；' +
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
      '<div class="qprogress"><span>先想 4 个问题，再往下看</span><span class="q-hint">' +
      Object.keys(state.quiz).length + " / " + data.length + " 已回答</span></div>" +
      '<div class="quiz">' + cards + "</div>"
    );
  }

  /* -------------------------------------------------------- composition */

  function composition(preset, c, m) {
    const perLayer = m.per_layer;
    return (
      '<h2 class="sec-title">1 · 模型由什么组成</h2>' +
      '<p class="page-lead">自回归 LM 只有四块：<b>embedding</b>（token id → 向量）、' +
      "<b>N 层 DecoderLayer</b>（每层是 attention + FFN/MoE）、<b>final norm</b>、<b>lm_head</b>（hidden → 词表）。" +
      "tokenizer 和采样器在模型外面。</p>" +
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
        c.experts ? "只读 top-" + c.top_k + " 的 expert；dense 模型这里等于总参数" : "dense：每步读全部") +
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
      '<h2 class="sec-title">2 · 为什么是多层？维度和层是什么关系</h2>' +
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
      '<h2 class="sec-title">3 · 参数如何参与一次 token 生成</h2>' +
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
      '<h2 class="sec-title">4 · GPU 如何加载参数</h2>' +
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
      '<h2 class="sec-title">5 · Token 生成调用了哪些 GPU 计算函数</h2>' +
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

  /* ---------------------------------------------------------------- notes */

  function thinkBoxes(notes) {
    return (
      '<h2 class="sec-title">6 · 用自己的话回答</h2>' +
      details("为什么层数增加，参数量和 KV 都线性增加，但 hidden 不变？", notes.layers + " " + notes.kv) +
      details("参数是什么？它和「知识」是什么关系？", notes.params) +
      details("模型、GPU、引擎在这套流程里各做了什么？", notes.gpu)
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
    el.querySelectorAll("[data-q]").forEach((b) =>
      b.addEventListener("click", () => {
        if (state.quiz[b.dataset.q] !== undefined) return;
        state.quiz[b.dataset.q] = +b.dataset.i;
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
    const ref = el.querySelector("#anatomy-reflect");
    if (ref) ref.addEventListener("click", () => go("reflect"));
  }
})();
