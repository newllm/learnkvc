/* Thinking-first views: prediction quiz, multi-ask simulator, attribution +
 * reflection. These views ask the reader to commit to a guess/answer before
 * revealing evidence. */
(function () {
  const LK = (window.LK = window.LK || {});
  const W = (LK.workshop = LK.workshop || {});
  const fmt = (n) => Number(n).toLocaleString("en-US");
  const esc = (s) =>
    String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const go = (view, step, stage) =>
    document.dispatchEvent(new CustomEvent("lk:navigate", { detail: { view, step, stage } }));

  const gib = (tokens, kib) => ((tokens * kib) / 1024 / 1024).toFixed(2);

  /* ============================================ progress + think tests */

  const PROGRESS_KEY = "learnkvc.progress.v1";
  const LEARNING_VIEWS = ["think", "kv", "prefix", "paged", "transformer", "anatomy", "train", "net", "nccl", "cuda", "ecn", "gpu", "npu", "opt"];
  const EMPTY_PROGRESS = { pre: null, post: null, gpu: "rtx4090", views: [], steps: [] };

  function loadProgress() {
    try {
      return Object.assign({}, EMPTY_PROGRESS, JSON.parse(localStorage.getItem(PROGRESS_KEY) || "{}"));
    } catch (e) {
      return Object.assign({}, EMPTY_PROGRESS);
    }
  }
  function saveProgress(p) {
    try {
      localStorage.setItem(PROGRESS_KEY, JSON.stringify(p));
    } catch (e) {
      /* file:// / private mode */
    }
  }
  LK.progress = {
    load: loadProgress,
    save: saveProgress,
    recordView(view) {
      const p = loadProgress();
      if (!p.views.includes(view)) {
        p.views.push(view);
        saveProgress(p);
      }
    },
    recordStep(id) {
      const p = loadProgress();
      if (!p.steps.includes(id)) {
        p.steps.push(id);
        saveProgress(p);
      }
    },
  };

  let draftPre = {};
  let draftPost = {};

  const calcP = (n) => {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return String(n);
  };
  const gb = (n) => (n >= 100 ? n.toFixed(0) : n.toFixed(1)) + " GB";

  function row(k, v) {
    return '<div class="lrow"><span>' + esc(k) + "</span><code>" + esc(v) + "</code></div>";
  }

  function csspec(k, v) {
    return '<div class="spec"><span>' + esc(k) + "</span><b>" + esc(v) + "</b></div>";
  }

  /* ------------------------------------------------- learning context */

  function contextCard(context, presets, p) {
    const preset = presets.find((x) => x.id === context.modelId) || presets[0];
    const m = preset.math;
    const gpu = context.gpus.find((g) => g.id === p.gpu) || context.gpus[0];
    const ctx = context.kvContext || 32768;
    const kvGib = (ctx * m.kv_bytes) / 1073741824;
    const need = m.weights_q4_gb + kvGib + 1.5;
    const tone = gpu.unified ? "bad" : gpu.vram >= need + 2 ? "ok" : gpu.vram >= need ? "warn" : "bad";
    const readBytes = m.active * (4.85 / 8) + ctx * m.kv_bytes;
    const decodeMax = gpu.bandwidth ? (gpu.bandwidth * 1e9) / readBytes : 0;
    const flops = 2 * m.active * 10000 + 4 * preset.layers * preset.q_heads * preset.head_dim * 1e8;
    const prefillMin = gpu.bf16 ? flops / (gpu.bf16 * 1e12) : 0;
    const fitText =
      tone === "ok"
        ? "显存宽裕，可以开更长上下文"
        : tone === "warn"
        ? "能跑，但只剩 1–2 GB，上下文和并发都要克制"
        : "放不下 / 会大量 offload，需要更小量化或更大显存";
    return (
      '<div class="learnctx">' +
      '<div class="ctx-title"><span>本次学习设定</span><b>' + esc(preset.name) + "</b></div>" +
      '<p class="note">' + esc(context.note) + "</p>" +
      '<div class="ctx-chips"><span>认识 GPU（GPU / NPU / 国产算力）</span><span>① KV cache 章</span>' +
      "<span>② Prefix 章（含多次提问）</span><span>③ PagedAttention 章</span>" +
      "<span>推理网络（NCCL / 无损）</span><span>认识 LLM（解剖 / 训练 / 计算器）</span>" +
      "<span>LLM 优化（FlashAttention / 量化 / 投机）</span></div>" +
      '<div class="ctx-grid">' +
      '<div class="ctx-model"><h4>LLM 信息</h4>' +
      row("总参数 / 激活", calcP(m.total) + " / " + calcP(m.active)) +
      row("层数 / hidden", preset.layers + " 层 / " + preset.hidden) +
      row("Q / KV 头", preset.q_heads + " / " + preset.kv_heads + "（head_dim " + preset.head_dim + "）") +
      row("FFN / Expert", preset.experts
        ? preset.experts + " expert（top-" + preset.top_k + "，宽 " + preset.expert_inter + "）"
        : "FFN " + preset.ffn) +
      row("词表 / 上下文", preset.vocab + " / " + preset.max_pos) +
      row("KV / token", (m.kv_bytes / 1024).toFixed(0) + " KiB") +
      row("权重 Q4 / BF16", gb(m.weights_q4_gb) + " / " + gb(m.weights_bf16_gb)) +
      "</div>" +
      '<div class="ctx-gpu"><h4>GPU 硬件（点击切换）</h4>' +
      '<div class="gpu-row">' +
      context.gpus
        .map(
          (g) =>
            '<button class="gpu' + (g.id === gpu.id ? " on" : "") + '" data-gpu="' + g.id + '">' +
            esc(g.name) + "</button>"
        )
        .join("") +
      "</div>" +
      '<div class="gpu-specs">' +
      csspec("显存", gpu.unified ? gpu.vram + " GB（统一内存）" : gpu.vram + " GB") +
      csspec("带宽", gpu.bandwidth + " GB/s") +
      csspec("BF16 算力", gpu.bf16 + " TFLOPS") +
      csspec("PCIe", gpu.pcie ? "≈" + gpu.pcie + " GB/s" : "—") +
      "</div>" +
      '<p class="fit ' + tone + '">' + esc(preset.name) + " Q4 " + gb(m.weights_q4_gb) +
      " + " + (ctx / 1024).toFixed(0) + "K KV " + kvGib.toFixed(2) + " GiB + 运行时 ≈ " +
      need.toFixed(1) + " GB → " + esc(fitText) + "</p>" +
      '<p class="note">decode 理论上限 ≈ <b>' + Math.round(decodeMax) + " tok/s</b>" +
      "（每步读激活权重 + 全部 KV）；10K prefill 计算下限 ≈ <b>" + prefillMin.toFixed(1) +
      " s</b>。实际通常再差 2–3 倍。</p>" +
      '<p class="note">' + esc(gpu.note) + "</p>" +
      "</div></div></div>"
    );
  }

  /* --------------------------------------------------------- test cards */

  function testCards(qs, answers, submitted) {
    return qs
      .map((q, i) => {
        const chosen = answers[q.id];
        const opts = q.options
          .map((opt, k) => {
            let cls = "qopt";
            if (chosen === k) cls += " chosen";
            if (submitted) {
              if (k === q.answer) cls += " correct";
              else if (chosen === k) cls += " wrong";
            }
            return (
              '<button class="' + cls + '" data-q="' + q.id + '" data-i="' + k + '"' +
              (submitted ? " disabled" : "") + ">" +
              String.fromCharCode(65 + k) + ". " + esc(opt) + "</button>"
            );
          })
          .join("");
        const why = submitted
          ? '<div class="qwhy"><b>' + (chosen === q.answer ? "✓" : "✗") + "</b> " + esc(q.why) + "</div>"
          : "";
        const summary = q.summary
          ? '<p class="qsummary">' + esc(q.summary) + "</p>"
          : "";
        const detail = q.detail
          ? '<details class="qdetail"><summary>背景 / 怎么想</summary><p>' + esc(q.detail) + "</p></details>"
          : "";
        return (
          '<article class="qcard' + (submitted ? " answered" : "") + '">' +
          '<div class="qhead"><span class="qtag">' + esc(q.tag) + "</span>" +
          '<span class="qnum">Q' + (i + 1) + "</span></div>" +
          '<p class="qtext">' + esc(q.q) + "</p>" + summary + detail +
          '<div class="qopts">' + opts + "</div>" + why + "</article>"
        );
      })
      .join("");
  }

  function scoreOf(qs, answers) {
    return qs.filter((q) => answers[q.id] === q.answer).length;
  }

  function bindTest(el, qs, kind, rerender) {
    const p = loadProgress();
    if (p[kind]) {
      const reset = el.querySelector("#" + kind + "-reset");
      if (reset) {
        reset.addEventListener("click", () => {
          const pp = loadProgress();
          pp[kind] = null;
          saveProgress(pp);
          if (kind === "pre") draftPre = {};
          else draftPost = {};
          rerender();
        });
      }
      return;
    }
    el.querySelectorAll("[data-q]").forEach((b) =>
      b.addEventListener("click", () => {
        const draft = kind === "pre" ? draftPre : draftPost;
        draft[b.dataset.q] = +b.dataset.i;
        rerender();
      })
    );
    const submit = el.querySelector("#" + kind + "-submit");
    if (submit) {
      submit.addEventListener("click", () => {
        const draft = kind === "pre" ? draftPre : draftPost;
        if (Object.keys(draft).length < qs.length) return;
        const pp = loadProgress();
        pp[kind] = { answers: Object.assign({}, draft), score: scoreOf(qs, draft), at: Date.now() };
        saveProgress(pp);
        rerender();
      });
    }
  }

  /* ------------------------------------------------------- pre-test view */

  W.renderThink = function (el, data) {
    const p = loadProgress();
    // 题目数量变化（例如 6 → 10）时，旧基线不再可比：作废旧分数。
    if (p.pre && Object.keys(p.pre.answers || {}).length !== data.quiz.length) {
      p.pre = null;
      saveProgress(p);
    }
    const submitted = !!p.pre;
    const answers = submitted ? p.pre.answers : draftPre;
    const n = data.quiz.length;
    const chosen = Object.keys(draftPre).length;

    const status = submitted
      ? "<span>已提交 · " + p.pre.score + " / " + n + "</span>"
      : "<span>已选 " + chosen + " / " + n + "</span>";
    const scoreBox = submitted
      ? '<div class="scorebox"><b>' + p.pre.score + " / " + n + "</b><span>" +
        (p.pre.score === n ? "全对。带着「为什么」去读，而不是找答案" : "记住选错的那几题；读完回来做后测") +
        "</span></div>"
      : "";
    const actions = submitted
      ? '<div class="test-actions"><button class="btn ghost" id="pre-reset">重做前测</button></div>'
      : '<div class="test-actions"><button class="btn primary" id="pre-submit" ' +
        (chosen < n ? "disabled" : "") + ">提交前测（" + chosen + " / " + n + "）</button>" +
        '<span class="tool-note">提交后才显示对错和解释；分数只用来和后测对照</span></div>';

    el.innerHTML =
      contextCard(data.context, data.presets, p) +
      '<div class="qprogress"><span>前测 · 凭直觉选，不查资料</span>' + status + "</div>" +
      '<div class="quiz">' + testCards(data.quiz, answers, submitted) + "</div>" +
      scoreBox + actions;

    bindTest(el, data.quiz, "pre", () => W.renderThink(el, data));
    el.querySelectorAll("[data-gpu]").forEach((b) =>
      b.addEventListener("click", () => {
        const pp = loadProgress();
        pp.gpu = b.dataset.gpu;
        saveProgress(pp);
        W.renderThink(el, data);
      })
    );
  };

  /* ================================================================ reuse */

  const reuseState = { mode: "turn", turns: 3, requests: 3 };

  W.renderReuse = function (el, reuse, opts = {}) {
    const compact = !!opts.compact;
    const head = compact
      ? '<div class="viz-head"><h3>多次提问 · 多轮 vs 多请求</h3>' +
        '<span class="sub">切换场景，看 prefill 与 KV 占用的变化</span></div>'
      : '<p class="kicker">一次提问 / 多次提问</p>' +
        '<h1 class="page-title">KV cache 到底在哪一步起作用？</h1>' +
        '<p class="page-lead">同一次提问（一次 prefill + 多步 decode）、同一会话的多次提问（多轮）、' +
        "多个请求的多次提问（共享前缀）——三种场景复用同一个缓存，但省的东西不一样。" +
        "先操作，再回答下面的问题。</p>";
    el.innerHTML =
      head +
      '<div class="mode-tabs">' +
      '<button data-mode="turn" class="' + (reuseState.mode === "turn" ? "on" : "") + '">同一会话 · 多轮</button>' +
      '<button data-mode="request" class="' + (reuseState.mode === "request" ? "on" : "") + '">多个请求 · 共享前缀</button>' +
      "</div>" +
      '<div id="reuse-body"></div>';

    el.querySelectorAll("[data-mode]").forEach((b) =>
      b.addEventListener("click", () => {
        reuseState.mode = b.dataset.mode;
        W.renderReuse(el, reuse, opts);
      })
    );
    const body = el.querySelector("#reuse-body");
    if (reuseState.mode === "turn") renderTurn(body, reuse, el, opts);
    else renderRequests(body, reuse, el, opts);
  };

  function stat(label, value, sub) {
    return (
      '<div class="stat"><div class="stat-label">' + label + "</div>" +
      '<div class="stat-value">' + value + "</div>" +
      (sub ? '<div class="stat-sub">' + sub + "</div>" : "") +
      "</div>"
    );
  }

  function bars(rows) {
    const max = Math.max(1, ...rows.map((r) => r.value));
    return (
      '<div class="bars">' +
      rows
        .map(
          (r) =>
            '<div class="barrow ' + (r.cls || "") + '">' +
            '<span class="barlabel">' + r.label + "</span>" +
            '<div class="bar"><i style="width:' + Math.max(2, (r.value / max) * 100) + '%"></i></div>' +
            '<span class="barvalue">' + fmt(r.value) + "</span>" +
            "</div>"
        )
        .join("") +
      "</div>"
    );
  }

  function renderTurn(body, reuse, root, opts) {
    const n = reuseState.turns;
    const turns = reuse.turns.slice(0, n);
    const nc = turns.reduce((a, t) => a + t.prefill_nocache, 0);
    const c = turns.reduce((a, t) => a + t.prefill_cache, 0);
    const saved = nc - c;
    const pct = nc ? Math.round((saved / nc) * 100) : 0;
    const last = turns[turns.length - 1];

    body.innerHTML =
      '<div class="reuse-toolbar"><div class="controls">' +
      '<button class="btn" data-act="add" ' + (n >= 5 ? "disabled" : "") + ">再问一次 +1</button>" +
      '<button class="btn ghost" data-act="reset">重置</button>' +
      '<span class="tool-note">第 ' + n + " 轮 · 每次提问新增 50 token，回答 100 token</span>" +
      "</div></div>" +
      '<div class="compare">' +
      stat("不缓存 · 累计 prefill", fmt(nc) + " tok", "每轮重算整段历史") +
      stat("有缓存 · 累计 prefill", fmt(c) + " tok", "第 1 轮 " + (reuse.system + reuse.user) +
        " tok，之后每轮 " + reuse.user + " tok") +
      stat("省下的 prefill", pct + "%", fmt(saved) + " tok 没有重算") +
      "</div>" +
      bars([
        { label: "不缓存", value: nc, cls: "warn" },
        { label: "缓存历史 K/V", value: c, cls: "good" },
      ]) +
      '<div class="convo">' +
      '<div class="bubble sys"><span>system</span>' + fmt(reuse.system) + " tok</div>" +
      turns
        .map(
          (t) =>
            '<div class="bubble user"><span>第 ' + t.i + " 问</span>" + fmt(reuse.user) + " tok</div>" +
            '<div class="bubble ai"><span>回答</span>' + fmt(reuse.answer) + " tok · 本轮 prefill " +
            (t.i === 1 ? reuse.system + reuse.user : reuse.user) + " tok</div>"
        )
        .join("") +
      "</div>" +
      '<p class="note">当前缓存里的历史：<b>' + fmt(last.kv_tokens) + " tok</b> ≈ " +
      gib(last.kv_tokens, reuse.token_kib) + " GiB（96 KiB/token）。" +
      "无论用不用 cache，这些历史 K/V 都要留在显存里 —— <b>多轮对话省的是重复计算，不省显存</b>。</p>" +
      thinkBox(
        "多轮对话里，缓存越攒越多，为什么还是划算？它省的到底是什么？",
        "省的是每一轮把历史重新 prefill 的计算。显存不省，反而随对话增长；" +
          "所以引擎还要做上下文压缩/淘汰——那是引擎的策略，不是 LLM 的数学。"
      ) +
      '<div class="reuse-actions"><button class="btn primary" data-go="request">换个场景：多个请求共享前缀 →</button></div>';

    bindReuseControls(body, reuse, root, opts, {
      add: () => { reuseState.turns = Math.min(5, n + 1); W.renderReuse(root, reuse, opts); },
      reset: () => { reuseState.turns = 1; W.renderReuse(root, reuse, opts); },
    });
  }

  function renderRequests(body, reuse, root, opts) {
    const n = reuseState.requests;
    const reqs = reuse.requests.slice(0, n);
    const ncPrefill = reqs.reduce((a, r) => a + r.prefill_nocache, 0);
    const cPrefill = reqs.reduce((a, r) => a + r.prefill_cache, 0);
    const ncKv = n * (reuse.system + reuse.question);
    const cKv = reuse.system + n * reuse.question;
    const prefillPct = Math.round(((ncPrefill - cPrefill) / ncPrefill) * 100);
    const kvPct = Math.round(((ncKv - cKv) / ncKv) * 100);

    body.innerHTML =
      '<div class="reuse-toolbar"><div class="controls">' +
      '<button class="btn" data-act="add" ' + (n >= 5 ? "disabled" : "") + ">新增请求 +1</button>" +
      '<button class="btn ghost" data-act="reset">重置</button>' +
      '<span class="tool-note">' + n + " 个请求 · system " + reuse.system +
      " tok 完全相同 · 每个问题 " + reuse.question + " tok</span></div></div>" +
      '<div class="compare">' +
      stat("累计 prefill · 不共享", fmt(ncPrefill) + " tok", "每个请求都重新算 system") +
      stat("累计 prefill · 共享前缀", fmt(cPrefill) + " tok", "system 只算一次，之后只算新问题") +
      stat("KV 占用 · 不共享", fmt(ncKv) + " tok", "≈ " + gib(ncKv, reuse.token_kib) + " GiB") +
      stat("KV 占用 · 共享", fmt(cKv) + " tok", "≈ " + gib(cKv, reuse.token_kib) + " GiB") +
      "</div>" +
      '<div class="sharedblocks">' +
      '<div class="block shared"><b>共享前缀块</b><span>' + fmt(reuse.system) + " tok · 引用计数 " + n + "</span></div>" +
      reqs
        .map(
          (r) =>
            '<div class="block private"><b>请求 ' + r.i + " 私有</b><span>" +
            fmt(reuse.question) + " tok</span></div>"
        )
        .join("") +
      "</div>" +
      bars([
        { label: "prefill 不共享", value: ncPrefill, cls: "warn" },
        { label: "prefill 共享", value: cPrefill, cls: "good" },
        { label: "KV 不共享", value: ncKv, cls: "warn" },
        { label: "KV 共享", value: cKv, cls: "good" },
      ]) +
      '<p class="note">省下 <b>' + prefillPct + "%</b> 的 prefill、<b>" + kvPct + "%</b> 的 KV 显存。" +
      "显存能省，是因为同一段前缀的块被 <b>引用计数共享</b>，只存一份；计算能省，是因为哈希命中了这段前缀。" +
      "当前共享前缀让 KV 从 " + fmt(ncKv) + " tok 降到 " + fmt(cKv) + " tok（system 只存一份，问题各自私有）。</p>" +
      thinkBox(
        "如果 3 个请求的 system prompt 前 190 token 一样、后 10 token 不同，还能共享多少？",
        "只能共享完全相同的那 190 token（按块粒度，可能更少）。前缀缓存匹配的是 token 前缀链，" +
          "从第一个不同的 token 开始，后面的 K/V 都不能复用——这是 common prefix 的定义，也是引擎按块做 hash 的原因。"
      ) +
      '<div class="reuse-actions"><button class="btn primary" data-go="turn">回到多轮对话 →</button></div>';

    bindReuseControls(body, reuse, root, opts, {
      add: () => { reuseState.requests = Math.min(5, n + 1); W.renderReuse(root, reuse, opts); },
      reset: () => { reuseState.requests = 1; W.renderReuse(root, reuse, opts); },
    });
  }

  function bindReuseControls(body, reuse, root, opts, fns) {
    body.querySelectorAll('[data-act="add"]').forEach((b) => b.addEventListener("click", fns.add));
    body.querySelectorAll('[data-act="reset"]').forEach((b) => b.addEventListener("click", fns.reset));
    body.querySelectorAll("[data-go]").forEach((b) =>
      b.addEventListener("click", () => {
        reuseState.mode = b.dataset.go;
        W.renderReuse(root, reuse, opts);
      })
    );
  }

  function thinkBox(q, a) {
    return (
      '<details class="thinkbox"><summary>想一想：' + esc(q) + "</summary>" +
      '<p>' + esc(a) + "</p></details>"
    );
  }

  /* ============================================================ feedback */

  W.renderFeedback = function (el, data) {
    const p = loadProgress();
    let pre = p.pre;
    let post = p.post;
    const qs = data.postTest.questions;
    const n = qs.length;
    // 题目数量变化时，旧前/后测分数作废，避免用旧分母展示。
    if (p.pre && Object.keys(p.pre.answers || {}).length !== data.preQuiz.length) {
      p.pre = null;
      pre = null;
      saveProgress(p);
    }
    if (p.post && Object.keys(p.post.answers || {}).length !== qs.length) {
      p.post = null;
      post = null;
      saveProgress(p);
    }
    const head =
      '<p class="kicker">学习反馈</p>' +
      '<h1 class="page-title">学完再答一遍</h1>' +
      '<p class="page-lead">' + esc(data.postTest.note) + "</p>";

    if (!post) {
      const count = Object.keys(draftPost).length;
      const preLine = pre
        ? '<p class="note">你的前测：<b>' + pre.score + " / " + n + "</b>。做完下面的后测，就能看到前后对照。</p>"
        : '<p class="note">还没做前测；后测只能单独看分数，无法对照。建议下次先做前测。</p>';
      el.innerHTML = head + preLine +
        '<div class="qprogress"><span>后测 · 已选 ' + count + " / " + n + "</span><span>不计时，凭理解选</span></div>" +
        '<div class="quiz">' + testCards(qs, draftPost, false) + "</div>" +
        '<div class="test-actions"><button class="btn primary" id="post-submit" ' +
        (count < n ? "disabled" : "") + ">提交后测（" + count + " / " + n + "）</button>" +
        '<span class="tool-note">提交后生成学习反馈</span></div>';
      bindTest(el, qs, "post", () => W.renderFeedback(el, data));
      return;
    }

    const delta = pre ? post.score - pre.score : null;
    const rows = data.preQuiz
      .map((preQ) => {
        const postQ = qs.find((q) => q.tag === preQ.tag) || qs[0];
        const preOk = pre ? pre.answers[preQ.id] === preQ.answer : null;
        const postOk = post.answers[postQ.id] === postQ.answer;
        let status = "维持";
        let cls = "ok";
        if (preOk === null) { status = "仅后测"; cls = ""; }
        else if (!preOk && postOk) { status = "提升"; cls = "good"; }
        else if (preOk && !postOk) { status = "退步"; cls = "bad"; }
        else if (!preOk && !postOk) { status = "仍需复习"; cls = "bad"; }
        return (
          "<tr><td>" + esc(preQ.tag) + "</td>" +
          "<td>" + (preOk === null ? "—" : preOk ? "✓" : "✗") + "</td>" +
          "<td>" + (postOk ? "✓" : "✗") + "</td>" +
          '<td class="' + cls + '">' + status + "</td></tr>"
        );
      })
      .join("");

    const wrong = qs.filter((q) => post.answers[q.id] !== q.answer);
    const suggestions = wrong
      .map((q) => {
        const sg = q.suggest || {};
        return (
          '<button class="btn" data-go="' + esc(sg.view || "kv") + '" data-step="' +
          (sg.step || 1) + '" data-stage="' + esc(sg.stage || "") + '">去复习：' +
          esc(q.tag) + " →</button>"
        );
      })
      .join("");
    const visitedViews = p.views.filter((v) => LEARNING_VIEWS.includes(v)).length;
    const totalSteps = data.totalSteps || 14;
    const doneSteps = p.steps.length;
    const when = (t) => (t ? new Date(t).toLocaleString() : "—");
    const lessonsContent = (window.LK_DATA && LK_DATA.lessons) || {};
    const chapterQuizRows = ALL_CHAPTERS.filter((ch) => (lessonsContent[ch] && (lessonsContent[ch].quiz || []).length))
      .map((ch) => {
        const qs = lessonsContent[ch].quiz;
        const l = (p.lessons || {})[ch] || {};
        const done = !!l.quizSubmitted;
        const sc = done ? (l.quizScore || 0) : null;
        return "<tr><td>" + esc(CHAPTER_LABELS[ch]) + "</td><td>" +
          (done ? sc + " / " + qs.length : "未做") + "</td><td>" +
          (done ? (sc === qs.length ? "全对" : "可回看") : "—") + "</td></tr>";
      })
      .join("");

    el.innerHTML = head +
      '<div class="compare">' +
      stat("前测", pre ? pre.score + " / " + n : "未做", pre ? when(pre.at) : "没有基线") +
      stat("后测", post.score + " / " + n, when(post.at)) +
      stat("变化", delta === null ? "—" : (delta >= 0 ? "+" : "") + delta,
        delta === null ? "无法比较" : delta > 0 ? "有提升" : delta === 0 ? "持平" : "退步，建议回看") +
      "</div>" +
      '<h3 class="sec-title">按概念对照</h3>' +
      '<div class="table-wrap"><table class="tagtable"><thead><tr>' +
      "<th>概念</th><th>前测</th><th>后测</th><th>状态</th></tr></thead><tbody>" +
      rows + "</tbody></table></div>" +
      '<h3 class="sec-title">学习完成度</h3>' +
      '<div class="completion">' +
      "<div><b>" + doneSteps + " / " + totalSteps + "</b><span>探索过的步骤</span></div>" +
      "<div><b>" + visitedViews + " / " + LEARNING_VIEWS.length + "</b><span>访问过的学习视图</span></div>" +
      "</div>" +
      (chapterQuizRows
        ? '<h3 class="sec-title">章节测验</h3>' +
          '<div class="table-wrap"><table class="tagtable"><thead><tr>' +
          "<th>章节</th><th>得分</th><th>状态</th></tr></thead><tbody>" + chapterQuizRows + "</tbody></table></div>"
        : "") +
      '<h3 class="sec-title">建议回看</h3>' +
      (suggestions
        ? '<div class="suggestions">' + suggestions + "</div>"
        : '<p class="note">后测全对，没有必须回看的概念。可以回到各章「总结」把理解写下来。</p>') +
      '<div class="summarycard"><b>概要</b><p>' +
      esc((data.summary && data.summary.brief) || data.takeaway || "") + "</p>" +
      (data.summary && data.summary.detail
        ? '<details><summary>详细描述</summary><p>' + esc(data.summary.detail) + "</p></details>"
        : "") +
      "</div>" +
      '<div class="test-actions"><button class="btn ghost" id="post-reset">重做后测</button>' +
      '<button class="btn ghost" data-go="think">回到前测 →</button></div>';

    bindTest(el, qs, "post", () => W.renderFeedback(el, data));
    el.querySelectorAll("[data-go]").forEach((b) =>
      b.addEventListener("click", () => {
        const step = b.dataset.step ? +b.dataset.step - 1 : undefined;
        go(b.dataset.go, step, b.dataset.stage || undefined);
      })
    );
  };

  /* ============================================================== reflect */

  const attrState = { assign: {}, submitted: false };
  const STORE_KEY = "learnkvc.reflect.v1";

  function loadNotes() {
    try {
      return JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
    } catch (e) {
      return {};
    }
  }

  function saveNotes(notes) {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(notes));
    } catch (e) {
      /* file:// private mode etc. — ignore */
    }
  }

  W.renderReflect = function (el, data) {
    const attr = data.attribution;
    const notes = loadNotes();
    const total = attr.cards.length;
    const assigned = Object.keys(attrState.assign).length;

    const bucketCols = attr.buckets
      .map((b) => {
        const items = attr.cards.filter((c) => attrState.assign[c.id] === b.id);
        return (
          '<div class="bucket" data-bucket="' + b.id + '">' +
          '<div class="bucket-head"><b>' + esc(b.label) + "</b><span>" + esc(b.desc) + "</span></div>" +
          '<div class="bucket-items">' +
          (items.length
            ? items.map((c) => '<span class="chip small">' + esc(c.label) + "</span>").join("")
            : '<span class="empty">把卡片分到这里</span>') +
          "</div></div>"
        );
      })
      .join("");

    const poolHTML = attrState.submitted ? "" : (attr.cards
      .map((c) => {
        const cur = attrState.assign[c.id];
        return (
          '<div class="acard" data-id="' + c.id + '">' +
          '<span class="alabel">' + esc(c.label) + "</span>" +
          '<span class="abtns">' +
          attr.buckets
            .map(
              (b) =>
                '<button class="abtn' + (cur === b.id ? " on" : "") +
                (attrState.submitted ? (c.bucket === b.id ? " ok" : cur === b.id ? " bad" : "") : "") +
                '" data-b="' + b.id + '">' + esc(b.label) + "</button>"
            )
            .join("") +
          "</span></div>"
        );
      })
      .join(""));

    const submitBtn = attrState.submitted
      ? ""
      : '<button class="btn primary" id="attr-submit" ' +
        (assigned < total ? "disabled" : "") + ">提交判断</button>";

    el.innerHTML =
      '<p class="kicker">归因 · LLM / GPU / 引擎</p>' +
      '<h1 class="page-title">这张卡片，是谁的工作？</h1>' +
      '<p class="page-lead">把 12 张卡片分到三层。分层不是绝对的；关键是说清「谁定义规则、谁执行、谁决定策略」。' +
      "分完点提交，会逐条给出理由。</p>" +
      '<div class="attr-progress" id="attr-progress"></div>' +
      '<div class="buckets">' + bucketCols + "</div>" +
      (poolHTML ? '<div class="cardpool">' + poolHTML + "</div>" : "") +
      '<div class="attr-actions">' + submitBtn +
      '<button class="btn ghost" id="attr-reset">重新归类</button>' +
      '<span class="tool-note" id="attr-score"></span></div>' +
      (attrState.submitted ? '<div class="attr-result">' + renderAttrResult(attr) + "</div>" : "") +
      "<h2 class='sec-title'>写下你的总结</h2>" +
      '<p class="page-lead">先写，再对照。写不完整没关系，能说出「一次 / 多次 / 三层」各一句就够了。</p>' +
      '<div class="rprompts">' +
      data.reflect.prompts
        .map(
          (p, i) =>
            '<div class="rprompt"><label for="r-' + p.id + '">' + (i + 1) + ". " + esc(p.q) + "</label>" +
            '<textarea id="r-' + p.id + '" data-id="' + p.id + '" rows="3">' +
            esc(notes[p.id] || "") + "</textarea></div>"
        )
        .join("") +
      "</div>" +
      '<div class="attr-actions"><button class="btn" id="notes-save">保存</button>' +
      '<button class="btn ghost" id="notes-copy">复制我的总结</button>' +
      '<button class="btn ghost" id="notes-ref">对照参考</button>' +
      '<span class="tool-note" id="notes-status"></span></div>' +
      '<div id="notes-refbox" hidden>' + renderReference(data.reflect) + "</div>";

    updateAttrUI(el, attr);

    // attribution handlers
    el.querySelectorAll(".acard").forEach((card) =>
      card.querySelectorAll(".abtn").forEach((btn) =>
        btn.addEventListener("click", () => {
          if (attrState.submitted) return;
          attrState.assign[card.dataset.id] = btn.dataset.b;
          W.renderReflect(el, data);
        })
      )
    );
    const submit = el.querySelector("#attr-submit");
    if (submit) {
      submit.addEventListener("click", () => {
        attrState.submitted = true;
        W.renderReflect(el, data);
      });
    }
    const reset = el.querySelector("#attr-reset");
    if (reset) {
      reset.addEventListener("click", () => {
        attrState.assign = {};
        attrState.submitted = false;
        W.renderReflect(el, data);
      });
    }

    // notes handlers
    el.querySelectorAll(".rprompt textarea").forEach((ta) =>
      ta.addEventListener("input", () => {
        const all = loadNotes();
        all[ta.dataset.id] = ta.value;
        saveNotes(all);
      })
    );
    el.querySelector("#notes-save").addEventListener("click", () => {
      const status = el.querySelector("#notes-status");
      status.textContent = "已保存到本地（localStorage）";
    });
    el.querySelector("#notes-copy").addEventListener("click", () => {
      const text = data.reflect.prompts
        .map((p, i) => i + 1 + ". " + p.q + "\n" + (loadNotes()[p.id] || "（还没写）"))
        .join("\n\n");
      copyText(text, el.querySelector("#notes-status"));
    });
    el.querySelector("#notes-ref").addEventListener("click", () => {
      const box = el.querySelector("#notes-refbox");
      box.hidden = !box.hidden;
    });
  };

  function updateAttrUI(el, attr) {
    const total = attr.cards.length;
    const assigned = Object.keys(attrState.assign).length;
    const progress = el.querySelector("#attr-progress");
    if (progress) {
      progress.innerHTML = "已归类 <b>" + assigned + "</b> / " + total +
        (attrState.submitted ? " · 已提交" : "");
    }
    const score = el.querySelector("#attr-score");
    if (score && attrState.submitted) {
      const right = attr.cards.filter((c) => attrState.assign[c.id] === c.bucket).length;
      score.textContent =
        right === total
          ? "命中 " + right + " / " + total + "。看下面的理由，确认边界情况。"
          : "命中 " + right + " / " + total + "。看下面的理由，重点看错的那几张。";
    }
  }

  function renderAttrResult(attr) {
    return attr.cards
      .map((c) => {
        const chosen = attrState.assign[c.id];
        const ok = chosen === c.bucket;
        const label = attr.buckets.find((b) => b.id === c.bucket).label;
        return (
          '<div class="aresult ' + (ok ? "ok" : "bad") + '">' +
          "<b>" + esc(c.label) + "</b> → " + esc(label) +
          (ok ? "" : "（你选了 " + esc(attr.buckets.find((b) => b.id === chosen).label) + "）") +
          '<div class="awhy">' + esc(c.why) + "</div></div>"
        );
      })
      .join("") +
      '<p class="note boundary">' + esc(attr.afterword) + "</p>";
  }

  function renderReference(reflect) {
    return (
      '<div class="reference">' +
      reflect.reference
        .map(
          (r) =>
            '<div class="refblock"><b>' + esc(r.title) + "</b><ul>" +
            r.points.map((p) => "<li>" + esc(p) + "</li>").join("") +
            "</ul></div>"
        )
        .join("") +
      '<p class="takeaway">' + esc(reflect.takeaway) + "</p></div>"
    );
  }

  function copyText(text, statusEl) {
    const done = () => (statusEl.textContent = "已复制");
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(() => fallbackCopy(text, done));
    } else {
      fallbackCopy(text, done);
    }
  }

  function fallbackCopy(text, done) {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
      done();
    } catch (e) {
      /* ignore */
    }
    document.body.removeChild(ta);
  }

  /* =================================================== lesson stages */

  const CHAPTER_LABELS = { transformer: "Transformer 与 LLM", kv: "单请求 KV cache", prefix: "Prefix cache", paged: "PagedAttention", net: "推理网络", train: "LLM 训练", nccl: "NCCL 学习章节", cuda: "CUDA 执行模型", ecn: "ECN + DCQCN", gpu: "认识 GPU", npu: "NPU 与国产算力", opt: "LLM 优化" };
  const ALL_CHAPTERS = ["transformer", "gpu", "npu", "cuda", "kv", "prefix", "paged", "net", "nccl", "ecn", "train", "opt"];

  function lessonOf(chapter) {
    const p = loadProgress();
    const base = {
      think: false,
      thinkAnswers: {},
      exploreMax: 0,
      reflect: false,
      reflectText: {},
      summary: false,
      summaryText: "",
      summaryAt: null,
      quizAnswers: {},
      quizScore: null,
      quizSubmitted: false,
    };
    return Object.assign(base, ((p.lessons || {})[chapter]) || {});
  }

  function saveLesson(chapter, patch) {
    const p = loadProgress();
    p.lessons = p.lessons || {};
    p.lessons[chapter] = Object.assign(lessonOf(chapter), patch);
    saveProgress(p);
  }

  LK.progress.lesson = lessonOf;
  LK.progress.saveLesson = saveLesson;

  const stageAnswers = {};

  W.renderStageThink = function (el, chapter, questions, onChange, onExplore) {
    const saved = lessonOf(chapter).thinkAnswers || {};
    stageAnswers[chapter] = Object.assign({}, saved);
    const n = questions.length;
    const answered = Object.keys(stageAnswers[chapter]).length;

    const cards = questions
      .map((q, i) => {
        const choice = stageAnswers[chapter][q.id || i];
        const done = choice !== undefined;
        const opts = q.options
          .map((opt, k) => {
            let cls = "qopt";
            if (done) {
              if (k === q.answer) cls += " correct";
              else if (choice === k) cls += " wrong";
            }
            return (
              '<button class="' + cls + '" data-stage-q="' + (q.id || i) + '" data-i="' + k + '"' +
              (done ? " disabled" : "") + ">" + String.fromCharCode(65 + k) + ". " + esc(opt) + "</button>"
            );
          })
          .join("");
        return (
          '<article class="qcard' + (done ? " answered" : "") + '">' +
          '<div class="qhead"><span class="qtag">请回答</span><span class="qnum">Q' + (i + 1) + "</span></div>" +
          '<p class="qtext">' + esc(q.q) + "</p>" +
          '<div class="qopts">' + opts + "</div>" +
          (done ? '<div class="qwhy"><b>' + (choice === q.answer ? "✓" : "✗") + "</b> " + esc(q.why) + "</div>" : "") +
          "</article>"
        );
      })
      .join("");

    const thinkTitle = chapter === "net" ? "先猜，再去看网络" : chapter === "train" ? "先猜，再算这笔账" : "先猜，再进代码";
    el.innerHTML =
      '<div class="doc-kicker">' + esc(CHAPTER_LABELS[chapter]) + " · 请回答</div>" +
      '<h2>' + thinkTitle + "</h2>" +
      '<p class="hint">不查资料，凭直觉选；选完才出现解释。请回答阶段不计分，目的是带着假设去探索。</p>' +
      '<div class="qprogress"><span>已想 ' + answered + " / " + n + "</span><span>" +
      (answered === n ? "可以进入探索了" : "答完再往下") + "</span></div>" +
      '<div class="quiz">' + cards + "</div>" +
      (answered === n
        ? '<div class="reuse-actions"><button class="btn primary" id="stage-explore">进入探索阅读 →</button></div>'
        : "");

    el.querySelectorAll("[data-stage-q]").forEach((b) =>
      b.addEventListener("click", () => {
        const qid = b.dataset.stageQ;
        if (stageAnswers[chapter][qid] !== undefined) return;
        stageAnswers[chapter][qid] = +b.dataset.i;
        const answers = Object.assign({}, stageAnswers[chapter]);
        const all = Object.keys(answers).length === n;
        saveLesson(chapter, { thinkAnswers: answers, think: all });
        W.renderStageThink(el, chapter, questions, onChange, onExplore);
        if (onChange) onChange();
      })
    );
    const go = el.querySelector("#stage-explore");
    if (go) go.addEventListener("click", onExplore);
  };

  W.renderStageReview = function (el, chapter, prompts, attribution, summary, onChange) {
    const p = lessonOf(chapter);
    const saved = p.reflectText || {};
    const promptList = prompts || [];
    const filled = promptList.filter((q, i) => (saved[q.id || i] || "").trim()).length;
    const attr = attribution || [];
    const attrDone = !attr.length || !!p.attributionSubmitted;
    const complete = !!p.summary;
    const content = ((window.LK_DATA && LK_DATA.lessons) || {})[chapter] || {};
    const quiz = content.quiz || [];
    const answers = p.quizAnswers || {};
    const submitted = !!p.quizSubmitted;
    const score = p.quizScore || 0;
    const answered = quiz.filter((q, i) => answers[i] !== undefined).length;
    const rerender = () => W.renderStageReview(el, chapter, prompts, attribution, summary, onChange);

    const quizHTML = quiz.length
      ? '<h3 class="sec-title">章节测验 · ' + quiz.length + " 题</h3>" +
        '<p class="hint">读完后用两题检查是否真的掌握；提交后显示得分与解释。</p>' +
        '<div class="quiz">' +
        quiz.map((q, i) => {
          const chosen = answers[i];
          const done = submitted;
          const opts = q.options.map((opt, k) => {
            let cls = "qopt";
            if (chosen === k) cls += " chosen";
            if (done) {
              if (k === q.answer) cls += " correct";
              else if (chosen === k) cls += " wrong";
            }
            return '<button class="' + cls + '" data-cq="' + i + '" data-ci="' + k + '"' +
              (done ? " disabled" : "") + ">" + String.fromCharCode(65 + k) + ". " + esc(opt) + "</button>";
          }).join("");
          return '<article class="qcard' + (done ? " answered" : "") + '">' +
            '<div class="qhead"><span class="qtag">章节测验</span><span class="qnum">Q' + (i + 1) + "</span></div>" +
            '<p class="qtext">' + esc(q.q) + "</p>" +
            '<div class="qopts">' + opts + "</div>" +
            (done ? '<div class="qwhy"><b>' + (chosen === q.answer ? "✓" : "✗") + "</b> " + esc(q.why || "") + "</div>" : "") +
            "</article>";
        }).join("") +
        "</div>" +
        '<div class="test-actions">' +
        (submitted
          ? '<span class="tool-note">得分 ' + score + " / " + quiz.length + '</span><button class="btn ghost" id="chapter-quiz-reset">重做章节测验</button>'
          : '<button class="btn primary" id="chapter-quiz-submit" ' + (answered < quiz.length ? "disabled" : "") +
            ">提交章节测验（" + answered + " / " + quiz.length + "）</button>" +
            '<span class="tool-note">提交后显示得分与解释</span>') +
        "</div>"
      : "";

    el.innerHTML =
      '<div class="doc-kicker">' + esc(CHAPTER_LABELS[chapter]) + " · 反思与总结</div>" +
      '<h2>先用自己的话回答，再判断这是谁的工作</h2>' +
      '<p class="hint">反思写不出来，说明探索还没闭环；归因把「LLM / GPU / 引擎」的分工说清楚；最后用三句话收束整章。参考默认折叠，先写再看。</p>' +
      '<div class="qprogress"><span>反思已写 ' + filled + " / " + promptList.length + "</span><span>保存到本地</span></div>" +
      promptList.map(
        (q, i) =>
          '<div class="rprompt"><label for="ref-' + (q.id || i) + '">' + (i + 1) + ". " + esc(q.q) + "</label>" +
          '<textarea id="ref-' + (q.id || i) + '" data-ref="' + (q.id || i) + '" rows="3">' + esc(saved[q.id || i] || "") + "</textarea>" +
          '<details class="thinkbox"><summary>对照参考</summary><p>' + esc(q.ref) + "</p></details></div>"
      ).join("") +
      '<h3 class="sec-title">归因：这一步是谁的工作？</h3>' +
      '<div id="stage-attr"></div>' +
      '<h3 class="sec-title">总结：参考 vs 我的版本</h3>' +
      '<div class="summarycard"><b>参考总结 · 概要</b><p>' + esc(summary.brief) + "</p>" +
      '<details><summary>详细描述</summary><p>' + esc(summary.detail) + "</p></details></div>" +
      '<div class="rprompt"><label for="stage-summary">我的总结</label>' +
      '<textarea id="stage-summary" rows="5">' + esc(p.summaryText || "") + "</textarea></div>" +
      quizHTML +
      '<div class="reuse-actions"><button class="btn primary" id="stage-review-done" ' +
      (attrDone ? "" : "disabled") + ">" +
      (complete ? "已完成的章 · 重新标记" : "完成本章 →") + "</button>" +
      '<button class="btn ghost" id="stage-export-md">导出 Markdown</button>' +
      (!attrDone
        ? '<span class="tool-note">先提交归因才能完成本章</span>'
        : complete
          ? '<span class="tool-note">完成于 ' + new Date(p.summaryAt).toLocaleString() + "</span>"
          : '<span class="tool-note">完成本章后会标记整章进度</span>') +
      "</div>";

    el.querySelectorAll("[data-ref]").forEach((ta) =>
      ta.addEventListener("input", () => {
        const prev = lessonOf(chapter);
        const all = Object.assign({}, prev.reflectText || {});
        all[ta.dataset.ref] = ta.value;
        saveLesson(chapter, { reflectText: all });
      })
    );
    W.renderStageAttribution(el.querySelector("#stage-attr"), chapter, attr, () => {
      rerender();
      if (onChange) onChange();
    });
    const ta = el.querySelector("#stage-summary");
    ta.addEventListener("input", () => saveLesson(chapter, { summaryText: ta.value }));
    el.querySelectorAll("[data-cq]").forEach((b) =>
      b.addEventListener("click", () => {
        if (submitted) return;
        const next = Object.assign({}, lessonOf(chapter).quizAnswers || {});
        next[b.dataset.cq] = +b.dataset.ci;
        saveLesson(chapter, { quizAnswers: next });
        if (onChange) onChange();
        rerender();
      })
    );
    const quizSubmit = el.querySelector("#chapter-quiz-submit");
    if (quizSubmit) {
      quizSubmit.addEventListener("click", () => {
        const finalAnswers = lessonOf(chapter).quizAnswers || {};
        const got = quiz.filter((q, i) => finalAnswers[i] === q.answer).length;
        saveLesson(chapter, { quizSubmitted: true, quizScore: got });
        if (onChange) onChange();
        rerender();
      });
    }
    const quizReset = el.querySelector("#chapter-quiz-reset");
    if (quizReset) {
      quizReset.addEventListener("click", () => {
        saveLesson(chapter, { quizAnswers: {}, quizSubmitted: false, quizScore: null });
        if (onChange) onChange();
        rerender();
      });
    }
    el.querySelector("#stage-review-done").addEventListener("click", () => {
      saveLesson(chapter, {
        reflect: true,
        summary: true,
        summaryText: ta.value,
        summaryAt: Date.now(),
      });
      if (onChange) onChange();
      rerender();
    });
    el.querySelector("#stage-export-md").addEventListener("click", () => {
      saveLesson(chapter, { summaryText: ta.value });
      exportChapterMarkdown(chapter);
    });
  };

  W.renderStageAttribution = function (el, chapter, cards, onChange) {
    if (!el) return;
    if (!cards || !cards.length) {
      el.innerHTML = "";
      return;
    }
    const lesson = lessonOf(chapter);
    const assigned = lesson.attribution || {};
    const submitted = !!lesson.attributionSubmitted;
    const buckets = [
      ["model", "LLM"],
      ["gpu", "GPU"],
      ["engine", "引擎"],
    ];
    const right = submitted
      ? cards.filter((c) => assigned[c.id] === c.bucket).length
      : 0;

    el.innerHTML =
      '<p class="hint">把下面 3 张卡分到「LLM / GPU / 引擎」；提交后逐条看理由。</p>' +
      cards
        .map((c) => {
          const cur = assigned[c.id];
          const btns = buckets
            .map(([id, label]) => {
              let cls = "abtn" + (cur === id ? " on" : "");
              if (submitted) {
                if (c.bucket === id) cls += " ok";
                else if (cur === id) cls += " bad";
              }
              return (
                '<button class="' + cls + '" data-attr="' + c.id + '" data-b="' + id + '"' +
                (submitted ? " disabled" : "") + ">" + label + "</button>"
              );
            })
            .join("");
          const why = submitted
            ? '<div class="awhy">' + (assigned[c.id] === c.bucket ? "✓ " : "✗ ") + esc(c.why) + "</div>"
            : "";
          return (
            '<div class="acard"' + (submitted ? ' style="align-items:flex-start"' : "") + ">" +
            '<span class="alabel">' + esc(c.label) + "</span>" +
            '<span class="abtns">' + btns + "</span>" + why + "</div>"
          );
        })
        .join("") +
      '<div class="attr-actions">' +
      (submitted
        ? '<button class="btn ghost" id="attr-reset">重新归类</button>' +
          '<span class="tool-note">命中 ' + right + " / " + cards.length + "</span>"
        : '<button class="btn primary" id="attr-submit" ' +
          (Object.keys(assigned).length < cards.length ? "disabled" : "") + ">提交判断</button>") +
      "</div>";

    el.querySelectorAll("[data-attr]").forEach((b) =>
      b.addEventListener("click", () => {
        if (submitted) return;
        const next = Object.assign({}, lessonOf(chapter).attribution || {});
        next[b.dataset.attr] = b.dataset.b;
        saveLesson(chapter, { attribution: next, attributionSubmitted: false });
        if (onChange) onChange();
      })
    );
    const submit = el.querySelector("#attr-submit");
    if (submit) {
      submit.addEventListener("click", () => {
        saveLesson(chapter, { attributionSubmitted: true });
        if (onChange) onChange();
      });
    }
    const reset = el.querySelector("#attr-reset");
    if (reset) {
      reset.addEventListener("click", () => {
        saveLesson(chapter, { attribution: {}, attributionSubmitted: false });
        if (onChange) onChange();
      });
    }
  };

  /* ======================================= chapter quiz / markdown export */

  function inlineMd(node) {
    if (!node) return "";
    if (node.nodeType === 3) return String(node.nodeValue).replace(/\s+/g, " ");
    if (node.nodeType !== 1) return "";
    const tag = node.tagName.toLowerCase();
    const inner = () => Array.from(node.childNodes).map(inlineMd).join("");
    if (tag === "strong" || tag === "b") return "**" + inner().trim() + "**";
    if (tag === "em" || tag === "i") return "*" + inner().trim() + "*";
    if (tag === "code") return "`" + node.textContent + "`";
    if (tag === "a") return "[" + inner().trim() + "](" + (node.getAttribute("href") || "") + ")";
    if (tag === "br") return "  \n";
    if (tag === "img") return "![" + (node.getAttribute("alt") || "") + "](" + (node.getAttribute("src") || "") + ")";
    return inner();
  }

  function blockMd(node, out) {
    if (!node) return;
    if (node.nodeType === 3) {
      const t = String(node.nodeValue).replace(/\s+/g, " ").trim();
      if (t) out.push(t);
      return;
    }
    if (node.nodeType !== 1) return;
    const tag = node.tagName.toLowerCase();
    if (["script", "style", "svg", "button", "select", "input"].includes(tag)) return;
    if (tag === "h1" || tag === "h2" || tag === "h3" || tag === "h4") {
      out.push("\n" + "#".repeat(+tag[1]) + " " + inlineMd(node).trim() + "\n");
      return;
    }
    if (tag === "p") {
      const t = inlineMd(node).trim();
      if (t) out.push(t + "\n");
      return;
    }
    if (tag === "ul" || tag === "ol") {
      out.push("");
      Array.from(node.children).forEach((li, i) => {
        if (li.tagName && li.tagName.toLowerCase() === "li") {
          out.push((tag === "ol" ? i + 1 + ". " : "- ") + inlineMd(li).trim());
        }
      });
      out.push("");
      return;
    }
    if (tag === "pre") {
      out.push("\n```\n" + node.textContent.replace(/\n+$/, "") + "\n```\n");
      return;
    }
    if (tag === "table") {
      const rows = Array.from(node.querySelectorAll("tr"));
      if (!rows.length) return;
      out.push("");
      rows.forEach((tr, ri) => {
        const cells = Array.from(tr.children).map((td) => inlineMd(td).trim().replace(/\|/g, "\\|"));
        out.push("| " + cells.join(" | ") + " |");
        if (ri === 0) out.push("| " + cells.map(() => "---").join(" | ") + " |");
      });
      out.push("");
      return;
    }
    if (tag === "blockquote") {
      const tmp = [];
      Array.from(node.childNodes).forEach((c) => blockMd(c, tmp));
      out.push(tmp.join("\n").split("\n").map((l) => (l ? "> " + l : ">")).join("\n"));
      return;
    }
    if (tag === "hr") {
      out.push("\n---\n");
      return;
    }
    if (tag === "details") {
      const sum = node.querySelector("summary");
      if (sum) out.push("\n**" + inlineMd(sum).trim() + "**\n");
      Array.from(node.childNodes).forEach((c) => { if (c !== sum) blockMd(c, out); });
      return;
    }
    Array.from(node.childNodes).forEach((c) => blockMd(c, out));
  }

  function htmlToMarkdown(html) {
    const box = document.createElement("div");
    box.innerHTML = html;
    const out = [];
    Array.from(box.childNodes).forEach((c) => blockMd(c, out));
    return out.join("\n").replace(/\n{3,}/g, "\n\n").trim();
  }

  function pruneExport(root) {
    root.querySelectorAll(
      "button,select,input,svg,script,style,.train-code,.train-run,.code-scroll,.sim-controls,.viz,.reuse-actions,.test-actions,.qopts,.stage-nav,.stepnav,.mode-tabs,.exprange"
    ).forEach((n) => n.remove());
  }

  function chapterContentMarkdown(chapter) {
    const data = window.LK_DATA && LK_DATA[chapter];
    if (!data) return "";
    if (data.steps) {
      return data.steps
        .map((s, i) => "### " + (i + 1) + ". " + s.title + "\n\n" + (s.hint ? "> " + s.hint + "\n\n" : "") + htmlToMarkdown(s.bodyHTML) + "\n")
        .join("\n");
    }
    const mod = window.LK && LK[chapter];
    if (mod && typeof mod.render === "function") {
      const holder = document.createElement("div");
      try {
        mod.render(holder, data, {
          presets: LK_DATA.anatomy.presets,
          gpus: LK_DATA.anatomy.context.gpus,
        });
        pruneExport(holder);
        return htmlToMarkdown(holder.innerHTML);
      } catch (e) {
        return "（本章内容导出失败，请在交互页查看完整教程。）";
      }
    }
    return "";
  }

  function downloadMarkdown(filename, text) {
    try {
      const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      /* file:// 或浏览器限制 */
    }
  }

  function exportChapterMarkdown(chapter) {
    const content = ((window.LK_DATA && LK_DATA.lessons) || {})[chapter] || {};
    const p = lessonOf(chapter);
    const label = CHAPTER_LABELS[chapter] || chapter;
    const parts = [
      "# " + label,
      "",
      "> 来自 learnkvc 交互式学习页 · " + new Date().toLocaleString(),
      "",
      "## 参考总结",
      "",
      (content.summary && content.summary.brief) || "",
      "",
      (content.summary && content.summary.detail) || "",
      "",
      "## 我的总结",
      "",
      (p.summaryText || "").trim() || "（还没写）",
      "",
      "## 反思与总结",
      "",
    ];
    (content.reflect || []).forEach((q) => {
      parts.push("- **" + q.q + "**");
      parts.push("  " + ((p.reflectText || {})[q.id] || "（未填写）"));
    });
    (content.attribution || []).forEach((c) => {
      parts.push("- " + c.label + " → " + ((p.attribution || {})[c.id] || "未归类"));
    });
    parts.push("");
    if ((content.quiz || []).length) {
      parts.push("## 章节测验", "");
      content.quiz.forEach((q, i) => {
        const chosen = (p.quizAnswers || {})[i];
        parts.push(i + 1 + ". " + q.q);
        q.options.forEach((o, k) => parts.push("   - " + String.fromCharCode(65 + k) + ". " + o));
        if (p.quizSubmitted) {
          parts.push(
            "   - 你的答案：" + (chosen === undefined ? "未选" : String.fromCharCode(65 + chosen)) +
            "；正确：" + String.fromCharCode(65 + q.answer) +
            "；" + (chosen === q.answer ? "✓" : "✗") + " " + (q.why || "")
          );
        }
        parts.push("");
      });
      if (p.quizSubmitted) parts.push("得分：" + (p.quizScore || 0) + " / " + content.quiz.length, "");
    }
    parts.push("## 本章内容", "", chapterContentMarkdown(chapter), "");
    downloadMarkdown("learnkvc-" + chapter + ".md", parts.join("\n"));
  }



})();
