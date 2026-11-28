/* Model training tutorial + per-GPU training-memory calculator.
 * Content lives in web/tools/train.json; the code panel is
 * example/train_tiny.py injected by build.py. */
(function () {
  const LK = (window.LK = window.LK || {});
  const T = (LK.train = LK.train || {});
  const fmt = (n) => Number(n).toLocaleString("en-US");
  const GB = 1e9;

  const state = {
    presetId: "qwen7b",
    gpu: "rtx4090",
    precision: "bf16",
    optimizer: "adamw",
    zero: 0,
    dp: 1,
    tp: 1,
    pp: 1,
    seq: 2048,
    batch: 4,
    ckpt: false,
    offload: false,
  };

  function gb(bytes) {
    return bytes / GB;
  }

  function gbfmt(value) {
    if (value >= 1024) return (value / 1024).toFixed(2) + " TB";
    if (value >= 10) return value.toFixed(1) + " GB";
    return value.toFixed(2) + " GB";
  }

  T.render = function (el, data, world) {
    const presets = world.presets;
    const gpus = world.gpus;
    const preset = presets.find((p) => p.id === state.presetId) || presets[0];
    const gpu = gpus.find((g) => g.id === state.gpu) || gpus[0];
    const N = preset.math.total;
    const wb = state.precision === "bf16" ? 2 : 4;
    const master = state.precision === "bf16" ? 4 : 0;
    const optimizerBytes = state.optimizer === "adamw" ? 8 + master : 4 + master;
    const totalGpus = state.dp * state.tp * state.pp;

    const wDiv = state.tp * state.pp * (state.zero === 3 ? state.dp : 1);
    const gDiv = state.tp * state.pp * (state.zero >= 2 ? state.dp : 1);
    const oDiv = state.tp * state.pp * (state.zero >= 1 ? state.dp : 1);
    const weights = (N * wb) / wDiv;
    const grads = (N * wb) / gDiv;
    const optimizer = (N * optimizerBytes) / oDiv;
    const layersPerStage = preset.layers / state.pp;
    const actFactor = state.ckpt ? 2.5 : 12;
    const act = (layersPerStage * state.batch * state.seq * preset.hidden * actFactor * wb) / state.tp;
    const overhead = 2 * GB;
    const onGpuOpt = state.offload ? 0 : optimizer;
    const total = weights + grads + onGpuOpt + act + overhead;
    const usable = gpu.vram * GB * (gpu.unified ? 0.75 : 0.9);
    const tone = total > usable ? "bad" : total > usable * 0.9 ? "warn" : "ok";
    const perSample =
      (layersPerStage * state.seq * preset.hidden * actFactor * wb) / state.tp;
    const fixed = weights + grads + onGpuOpt + overhead;
    const headroom = usable - fixed;
    const maxBatch = headroom > 0 ? Math.floor(headroom / Math.max(perSample, 1)) : 0;
    const batchNote =
      headroom <= 0
        ? "参数状态 + 预留已经超过单卡可用显存，先把 ZeRO-3 / FSDP、TP/PP 或 offload 打开，再谈 micro-batch。"
        : "当前估算最多约 <b>" + maxBatch + "</b> 个 micro-batch（受显存限制）；" +
          (maxBatch < state.batch ? "小于你选的 " + state.batch + "，先开激活重算或减小序列。" : "") +
          " global batch = micro-batch × 梯度累积 × DP。";

    const bar = (label, bytes, cls) =>
      '<div class="memrow ' + cls + '"><span>' + label + "</span>" +
      '<div class="membar"><i style="width:' + Math.max(1, (bytes / total) * 100) + '%"></i></div>' +
      "<b>" + gbfmt(gb(bytes)) + "</b></div>";

    const formula =
      "N = " + fmt(N) + "；权重 = N × " + wb + " B / (TP " + state.tp + " × PP " + state.pp +
      (state.zero === 3 ? " × ZeRO-3 DP " + state.dp : "") + ") = " + gbfmt(gb(weights)) + "；" +
      "梯度 = " + gbfmt(gb(grads)) + "；优化器 = " + (state.offload ? "已 offload 到 CPU" : gbfmt(gb(optimizer))) + "；" +
      "激活 = " + (preset.layers / state.pp).toFixed(0) + " 层 × batch " + state.batch + " × seq " +
      fmt(state.seq) + " × hidden " + fmt(preset.hidden) + " × " + actFactor + " × " + wb + " B / TP " +
      state.tp + " = " + gbfmt(gb(act)) + "。";

    el.innerHTML =
      '<p class="kicker">' + esc(data.kicker) + "</p>" +
      '<h1 class="page-title">' + esc(data.title) + "</h1>" +
      '<p class="page-lead">' + esc(data.lead) + "</p>" +

      /* 1. step */
      '<h2 class="sec-title">' + esc(data.stepSection.title) + "</h2>" +
      '<p class="page-lead">' + esc(data.stepSection.lead) + "</p>" +
      '<div class="train-flow">' +
      data.stepSection.steps
        .map(
          (s, i) =>
            '<div class="train-step"><span class="train-no">0' + (i + 1) + "</span>" +
            "<b>" + esc(s.name) + "</b><p>" + esc(s.what) + "</p>" +
            '<span class="train-keeps">' + esc(s.keeps) + "</span></div>"
        )
        .join("") +
      "</div>" +

      /* 2. ledger + calculator */
      '<h2 class="sec-title">' + esc(data.ledgerSection.title) + "</h2>" +
      '<p class="page-lead">' + esc(data.ledgerSection.lead) + "</p>" +
      '<div class="train-ledger">' +
      data.ledgerSection.items
        .map(
          (x) =>
            '<div class="train-ledger-card"><b>' + esc(x.title) + "</b>" +
            '<code>' + esc(x.formula) + "</code><p>" + esc(x.desc) + "</p></div>"
        )
        .join("") +
      "</div>" +
      '<ul class="train-rules">' + data.ledgerSection.rules.map((r) => "<li>" + esc(r) + "</li>").join("") + "</ul>" +
      '<h3 class="sec-sub">2.1 交互计算器：每张卡要装多少</h3>' +
      '<p class="note">' + esc(data.ledgerSection.calcLead) + "</p>" +
      '<div class="calc-layout train-calc">' +
      '<div class="calc-controls">' +
      '<label class="calc-field"><span>LLM</span><select data-field="preset">' +
      presets.map((p) => option(p.id, p.name, state.presetId)).join("") +
      "</select></label>" +
      '<label class="calc-field"><span>权重/梯度精度</span><select data-field="precision">' +
      option("bf16", "BF16（2 B）", state.precision) + option("fp32", "FP32（4 B）", state.precision) +
      "</select></label>" +
      '<label class="calc-field"><span>优化器</span><select data-field="optimizer">' +
      option("adamw", "AdamW（m + v）", state.optimizer) + option("sgd", "SGD + momentum", state.optimizer) +
      "</select></label>" +
      '<label class="calc-field"><span>ZeRO / FSDP</span><select data-field="zero">' +
      [0, 1, 2, 3]
        .map((z) =>
          option(z, z === 0 ? "ZeRO-0（不切）" : z === 3 ? "ZeRO-3 / FSDP（切参数）" : "ZeRO-" + z, state.zero)
        )
        .join("") +
      "</select></label>" +
      '<div class="train-par-row">' +
      '<label class="calc-field"><span>DP</span><select data-field="dp">' + nums([1, 2, 4, 8], state.dp) + "</select></label>" +
      '<label class="calc-field"><span>TP</span><select data-field="tp">' + nums([1, 2, 4, 8], state.tp) + "</select></label>" +
      '<label class="calc-field"><span>PP</span><select data-field="pp">' + nums([1, 2, 4, 8], state.pp) + "</select></label>" +
      "</div>" +
      '<label class="calc-field"><span>序列长度 <b>' + fmt(state.seq) + " tok</b></span>" +
      '<input type="range" data-field="seq" min="512" max="8192" step="512" value="' + state.seq + '"></label>' +
      '<label class="calc-field"><span>micro-batch <b>' + state.batch + "</b></span>" +
      '<input type="range" data-field="batch" min="1" max="16" step="1" value="' + state.batch + '"></label>' +
      '<label class="train-check"><input type="checkbox" data-field="ckpt"' + (state.ckpt ? " checked" : "") +
      "> 激活重算（checkpointing）</label>" +
      '<label class="train-check"><input type="checkbox" data-field="offload"' + (state.offload ? " checked" : "") +
      "> 优化器 offload 到 CPU</label>" +
      '<div class="calc-field"><span>GPU</span><div class="gpu-row">' +
      gpus
        .map(
          (g) =>
            '<button class="gpu' + (g.id === gpu.id ? " on" : "") + '" data-gpu="' + g.id + '">' +
            esc(g.name) + "</button>"
        )
        .join("") +
      "</div></div>" +
      '<div class="gpu-specs">' +
      '<div class="spec"><span>显存</span><b>' + gpu.vram + " GB" + (gpu.unified ? "（统一）" : "") + "</b></div>" +
      '<div class="spec"><span>训练可用</span><b>' + gbfmt(usable / GB) + "</b></div>" +
      '<div class="spec"><span>总卡数</span><b>DP ' + state.dp + " × TP " + state.tp + " × PP " + state.pp +
      " = " + totalGpus + "</b></div>" +
      "</div></div>" +

      '<div class="calc-result">' +
      '<div class="compare">' +
      '<div class="stat"><div class="stat-label">每卡峰值</div><div class="stat-value ' + tone + '">' +
      gbfmt(gb(total)) + '</div><div class="stat-sub">参数状态 ' + gbfmt(gb(weights + grads + onGpuOpt)) +
      " + 激活 " + gbfmt(gb(act)) + " + 预留 2 GB</div></div>" +
      '<div class="stat"><div class="stat-label">放得下吗</div><div class="stat-value ' + tone + '">' +
      (tone === "ok" ? "宽裕" : tone === "warn" ? "紧张" : "放不下") +
      '</div><div class="stat-sub">可用预算 ≈ ' + gbfmt(usable / GB) +
      "（显存 × " + (gpu.unified ? "75%" : "90%") + "）</div></div>" +
      "</div>" +
      '<div class="membars">' +
      bar("权重", weights, "w") +
      bar("梯度", grads, "k") +
      bar("优化器", onGpuOpt, "r") +
      bar("激活", act, "a") +
      bar("预留", overhead, "r") +
      "</div>" +
      '<p class="note train-formula">' + esc(formula) + "</p>" +
      '<p class="note">' + batchNote +
      (state.offload && optimizer > 0
        ? " offload 的优化器状态需要额外 CPU 内存 ≈ " + gbfmt(gb(optimizer)) + "。"
        : "") +
      "</p>" +
      '<p class="note">' + esc(data.ledgerSection.calcNote) + "</p>" +
      "</div></div>" +

      /* 3. parallel */
      '<h2 class="sec-title">' + esc(data.parallelSection.title) + "</h2>" +
      '<p class="page-lead">' + esc(data.parallelSection.lead) + "</p>" +
      '<div class="table-wrap"><table class="tagtable train-par-table"><thead><tr>' +
      "<th>方式</th><th>切什么</th><th>通信 / 代价</th><th>什么时候用</th>" +
      "</tr></thead><tbody>" +
      data.parallelSection.rows
        .map(
          (r) =>
            "<tr><td><b>" + esc(r.name) + "</b></td><td>" + esc(r.shards) + "</td><td>" +
            esc(r.comm) + "</td><td>" + esc(r.when) + "</td></tr>"
        )
        .join("") +
      "</tbody></table></div>" +
      '<ul class="train-rules">' + data.parallelSection.savings.map((s) => "<li>" + esc(s) + "</li>").join("") + "</ul>" +

      /* 4. script */
      '<h2 class="sec-title">' + esc(data.scriptSection.title) + "</h2>" +
      '<p class="page-lead">' + esc(data.scriptSection.lead) + "</p>" +
      '<pre class="train-run"><code>' + esc(data.scriptSection.run) + "</code></pre>" +
      '<ul class="train-rules">' + data.scriptSection.notes.map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul>" +
      '<div class="train-code-head"><span>' + esc(data.scriptSection.file) + "</span><span>PyTorch · CPU / GPU</span></div>" +
      '<div class="train-code" id="train-code"></div>' +

      /* 5. simulation code */
      '<h2 class="sec-title">' + esc(data.codeSection.title) + "</h2>" +
      '<p class="page-lead">' + esc(data.codeSection.lead) + "</p>" +
      '<pre class="train-run"><code>' + esc(data.codeSection.run) + "</code></pre>" +
      '<ul class="train-rules">' + data.codeSection.notes.map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul>" +
      '<div class="train-code-head"><span>' + esc(data.codeSection.file) + "</span><span>纯 Python · 可运行</span></div>" +
      '<div class="train-code" id="train-sim-code"></div>' +
      '<h3 class="sec-sub">示例输出</h3>' +
      '<pre class="train-run"><code>' + esc(data.simOutput) + "</code></pre>" +

      /* 6. check */
      '<h2 class="sec-title">' + esc(data.checkSection.title) + "</h2>" +
      '<p class="page-lead">' + esc(data.checkSection.lead) + "</p>" +
      '<ul class="train-rules train-checklist">' + data.checkSection.checklist.map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul>" +
      data.checkSection.questions
        .map((q) => details(q.q, q.a))
        .join("");

    const codeEl = el.querySelector("#train-code");
    if (codeEl && LK.codeview && data.code) LK.codeview.mount(codeEl, data.code.lines);
    const simCodeEl = el.querySelector("#train-sim-code");
    if (simCodeEl && LK.codeview && data.simCode) LK.codeview.mount(simCodeEl, data.simCode.lines);

    el.querySelectorAll("[data-field]").forEach((inp) => {
      inp.addEventListener("change", () => {
        const f = inp.dataset.field;
        if (inp.type === "checkbox") state[f] = inp.checked;
        else if (["zero", "dp", "tp", "pp", "seq", "batch"].includes(f)) state[f] = +inp.value;
        else state[f] = inp.value;
        T.render(el, data, world);
      });
    });
    el.querySelectorAll("[data-gpu]").forEach((b) =>
      b.addEventListener("click", () => {
        state.gpu = b.dataset.gpu;
        T.render(el, data, world);
      })
    );
  };

  function option(value, label, current) {
    return '<option value="' + esc(value) + '"' + (String(value) === String(current) ? " selected" : "") +
      ">" + esc(label) + "</option>";
  }

  function nums(list, current) {
    return list.map((n) => option(n, n, current)).join("");
  }

  function details(q, a) {
    return '<details class="thinkbox"><summary>' + esc(q) + "</summary><p>" + esc(a) + "</p></details>";
  }

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
})();
