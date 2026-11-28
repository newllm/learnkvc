/* Inference network: NCCL ring simulator, RoCE notes, PD separation calculator. */
(function () {
  const LK = (window.LK = window.LK || {});
  const N = (LK.net = LK.net || {});
  const fmt = (n) => Number(n).toLocaleString("en-US");

  const sim = {
    op: "allreduce",
    n: 8,
    sizeMb: 128,
    bwGbps: 400,
    latencyUs: 2,
    step: 0,
    playing: false,
    timer: null,
  };

  const pd = {
    presetId: "qwen30b",
    gpu: "rtx4090",
    prompt: 4096,
    output: 128,
    kv: "fp16",
    bwGbps: 400,
    latencyUs: 5,
    prefillGpus: 1,
    decodeGpus: 1,
  };

  const OPS = {
    allreduce: "AllReduce（ring）",
    allgather: "AllGather（ring）",
    reducescatter: "ReduceScatter（ring）",
    alltoall: "AllToAll（ring）",
    sendrecv: "Send / Recv（P2P）",
  };

  N.render = function (el, data, world) {
    stopSim();
    el.innerHTML =
      '<p class="kicker">' + esc(data.kicker) + "</p>" +
      '<h1 class="page-title">' + esc(data.title) + "</h1>" +
      '<p class="page-lead">' + esc(data.lead) + "</p>" +

      section(data.why, function () {
        return '<div class="net-cards">' +
          data.why.items.map((x, i) =>
            '<div class="net-card"><span class="train-no">0' + (i + 1) + "</span><b>" + esc(x.name) +
            "</b><p>" + esc(x.data) + '</p><span class="net-when">' + esc(x.when) + "</span></div>"
          ).join("") + "</div>";
      }) +

      section(data.nccl, function () {
        return '<div class="table-wrap"><table class="tagtable net-table"><thead><tr>' +
          "<th>Collective</th><th>语义</th><th>流量 / 步数</th><th>推理里的用途</th>" +
          "</tr></thead><tbody>" +
          data.nccl.collectives.map((c) =>
            "<tr><td><b>" + esc(c.name) + "</b></td><td>" + esc(c.what) + "</td><td>" +
            esc(c.bytes) + "</td><td>" + esc(c.use) + "</td></tr>"
          ).join("") + "</tbody></table></div>" +
          '<div class="net-cards two">' +
          data.nccl.algorithms.map((a) =>
            '<div class="net-card"><b>' + esc(a.name) + "</b><p>" + esc(a.tradeoff) +
            '</p><span class="net-when">' + esc(a.when) + "</span></div>"
          ).join("") + "</div>" +
          '<div class="table-wrap"><table class="tagtable net-table"><thead><tr>' +
          "<th>传输</th><th>带宽</th><th>延迟</th><th>范围 / 备注</th></tr></thead><tbody>" +
          data.nccl.transports.map((t) =>
            "<tr><td><b>" + esc(t.link) + "</b></td><td>" + esc(t.bw) + "</td><td>" +
            esc(t.latency) + "</td><td>" + esc(t.scope) + "</td></tr>"
          ).join("") + "</tbody></table></div>";
      }) +

      '<h2 class="sec-title">' + esc(data.sim.title) + "</h2>" +
      '<p class="page-lead">' + esc(data.sim.lead) + "</p>" +
      '<div id="nccl-sim" class="sim"></div>' +
      '<p class="note">' + esc(data.sim.note) + "</p>" +

      section(data.roce, function () {
        return '<ul class="train-rules">' + data.roce.concepts.map((c) => "<li>" + esc(c) + "</li>").join("") + "</ul>" +
          '<h3 class="sec-sub">NCCL / RoCE 常用环境变量</h3>' +
          '<div class="table-wrap"><table class="tagtable net-table envtable"><thead><tr>' +
          "<th>变量</th><th>示例</th><th>作用 / 排查点</th></tr></thead><tbody>" +
          data.roce.env.map((e) =>
            "<tr><td><code>" + esc(e.name) + "</code></td><td><code>" + esc(e.value) +
            "</code></td><td>" + esc(e.why) + "</td></tr>"
          ).join("") + "</tbody></table></div>" +
          '<h3 class="sec-sub">可以直接跑的命令</h3>' +
          '<pre class="train-run"><code>' + esc(data.roce.commands.join("\n")) + "</code></pre>" +
          '<div class="fncaveats"><b>常见坑</b><ul>' +
          data.roce.pitfalls.map((p) => "<li>" + esc(p) + "</li>").join("") + "</ul></div>";
      }) +

      '<h2 class="sec-title">' + esc(data.pd.title) + "</h2>" +
      '<p class="page-lead">' + esc(data.pd.lead) + "</p>" +
      '<div class="pd-flow">' + data.pd.flow.map((f, i) =>
        '<div class="pd-step"><span class="train-no">0' + (i + 1) + "</span><span>" + esc(f) + "</span></div>"
      ).join("") + "</div>" +
      '<div id="pd-sim" class="pd"></div>' +
      '<ul class="train-rules">' + data.pd.tradeoffs.map((t) => "<li>" + esc(t) + "</li>").join("") + "</ul>" +
      '<p class="note">' + esc(data.pd.note) + "</p>" +

      '<h2 class="sec-title">' + esc(data.codeSection.title) + "</h2>" +
      '<p class="page-lead">' + esc(data.codeSection.lead) + "</p>" +
      '<pre class="train-run"><code>' + esc(data.codeSection.run) + "</code></pre>" +
      '<ul class="train-rules">' + data.codeSection.notes.map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul>" +
      '<div class="train-code-head"><span>' + esc(data.codeSection.file) + "</span><span>纯 Python · 可运行</span></div>" +
      '<div class="train-code" id="net-sim-code"></div>' +
      '<h3 class="sec-sub">示例输出</h3>' +
      '<pre class="train-run"><code>' + esc(data.simOutput) + "</code></pre>" +

      section(data.check, function () {
        return '<ul class="train-rules train-checklist">' + data.check.checklist.map((c) => "<li>" + esc(c) + "</li>").join("") + "</ul>" +
          data.check.questions.map((q) => details(q.q, q.a)).join("");
      });

    mountSim(el.querySelector("#nccl-sim"), data);
    mountPd(el.querySelector("#pd-sim"), data, world);
    const simCodeEl = el.querySelector("#net-sim-code");
    if (simCodeEl && LK.codeview && data.simCode) LK.codeview.mount(simCodeEl, data.simCode.lines);
  };

  /* --------------------------------------------------------------- sim */

  function mountSim(host, data) {
    if (!host) return;
    host.innerHTML =
      '<div class="sim-controls">' +
      control("操作", '<select data-sim="op">' +
        Object.keys(OPS).map((k) => '<option value="' + k + '"' + (sim.op === k ? " selected" : "") + ">" + OPS[k] + "</option>").join("") +
        "</select>") +
      control("Rank 数", '<select data-sim="n">' +
        [2, 4, 8, 16].map((v) => option(v, v + " 卡", sim.n)).join("") + "</select>") +
      control("消息大小 S", '<select data-sim="size">' +
        [1, 16, 128, 512].map((v) => option(v, v + " MiB", sim.sizeMb)).join("") + "</select>") +
      control("链路带宽", '<select data-sim="bw">' +
        [100, 200, 400, 800].map((v) => option(v, v + " Gb/s（" + v / 8 + " GB/s）", sim.bwGbps)).join("") + "</select>") +
      control("单跳延迟", '<select data-sim="lat">' +
        [1, 2, 5, 10].map((v) => option(v, v + " µs", sim.latencyUs)).join("") + "</select>") +
      "</div>" +
      '<div class="sim-body"><div class="sim-ring" id="sim-ring"></div>' +
      '<div class="sim-side"><div class="sim-phase" id="sim-phase"></div>' +
      '<div class="sim-metrics" id="sim-metrics"></div></div></div>' +
      '<div class="sim-stepbar">' +
      '<button class="nav-btn" data-sim-step="-1" title="上一步">←</button>' +
      '<button class="btn" id="sim-play">▶ 播放</button>' +
      '<button class="nav-btn" data-sim-step="1" title="下一步">→</button>' +
      '<span id="sim-step-label"></span>' +
      '<div class="sim-progress"><i id="sim-progress"></i></div>' +
      '<button class="btn ghost" id="sim-reset">重置</button>' +
      "</div>" +
      '<div class="sim-chunk-wrap"><table class="sim-chunk" id="sim-chunk"></table></div>';

    host.querySelectorAll("[data-sim]").forEach((inp) => {
      inp.addEventListener("change", () => {
        const f = inp.dataset.sim;
        if (f === "op") sim.op = inp.value;
        else if (f === "n") sim.n = +inp.value;
        else if (f === "size") sim.sizeMb = +inp.value;
        else if (f === "bw") sim.bwGbps = +inp.value;
        else if (f === "lat") sim.latencyUs = +inp.value;
        stopSim();
        sim.step = 0;
        updateSim(host);
      });
    });
    host.querySelectorAll("[data-sim-step]").forEach((b) =>
      b.addEventListener("click", () => {
        stopSim();
        sim.step = clamp(sim.step + +b.dataset.simStep, 0, totalSteps() - 1);
        updateSim(host);
      })
    );
    const play = host.querySelector("#sim-play");
    play.addEventListener("click", () => {
      if (sim.playing) stopSim();
      else startSim(host);
      updateSim(host);
    });
    host.querySelector("#sim-reset").addEventListener("click", () => {
      stopSim();
      sim.step = 0;
      updateSim(host);
    });
    updateSim(host);
  }

  function startSim(host) {
    if (sim.step >= totalSteps() - 1) sim.step = 0;
    sim.playing = true;
    sim.timer = setInterval(() => {
      if (sim.step >= totalSteps() - 1) {
        stopSim();
        updateSim(host);
        return;
      }
      sim.step += 1;
      updateSim(host);
    }, 700);
  }

  function stopSim() {
    sim.playing = false;
    if (sim.timer) clearInterval(sim.timer);
    sim.timer = null;
  }

  function totalSteps() {
    const n = sim.n;
    if (sim.op === "sendrecv") return 1;
    if (sim.op === "allreduce") return 2 * (n - 1);
    return n - 1;
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function stepInfo() {
    const n = sim.n;
    const s = sim.step;
    const isAllReduce = sim.op === "allreduce";
    let phase = "";
    let localStep = s;
    if (isAllReduce) {
      if (s < n - 1) phase = "reduce-scatter";
      else { phase = "all-gather"; localStep = s - (n - 1); }
    }
    const rows = [];
    for (let r = 0; r < n; r++) {
      let send = "—";
      let to = "—";
      if (sim.op === "sendrecv") {
        if (r === 0) { send = "整个 S"; to = "rank 1"; }
      } else if (sim.op === "alltoall") {
        const dest = (r + s + 1) % n;
        send = "发往 rank " + dest + " 的 S/" + n;
        to = "rank " + ((r + 1) % n);
      } else {
        let chunk;
        if (isAllReduce && phase === "all-gather") chunk = (r - localStep + 1 + n) % n;
        else chunk = (r - localStep + n) % n;
        send = "chunk " + chunk + "（S/" + n + "）";
        to = "rank " + ((r + 1) % n);
      }
      rows.push({ r, send, to });
    }
    const stepName = sim.op === "sendrecv"
      ? "rank 0 → rank 1"
      : sim.op === "alltoall"
        ? "每个 rank 把「发往 (r + " + (sim.step + 1) + ") mod " + n + "」的 chunk 发给下一跳"
        : phase + " 第 " + (localStep + 1) + " / " + (totalSteps() > 1 && isAllReduce ? n - 1 : totalSteps()) + " 步";
    return { phase, rows, stepName };
  }

  function updateSim(host) {
    if (!host) return;
    const info = stepInfo();
    const label = host.querySelector("#sim-step-label");
    if (label) label.textContent = "step " + (sim.step + 1) + " / " + totalSteps();
    const prog = host.querySelector("#sim-progress");
    if (prog) prog.style.width = ((sim.step + 1) / totalSteps() * 100).toFixed(1) + "%";
    const play = host.querySelector("#sim-play");
    if (play) play.textContent = sim.playing ? "⏸ 暂停" : sim.step >= totalSteps() - 1 ? "↻ 重播" : "▶ 播放";

    const phase = host.querySelector("#sim-phase");
    if (phase) {
      phase.innerHTML = "<b>" + esc(OPS[sim.op]) + "</b><span>" + esc(info.stepName) + "</span>";
    }

    const ring = host.querySelector("#sim-ring");
    if (ring) ring.innerHTML = ringSvg(sim.n, sim.op, sim.step);

    const metrics = host.querySelector("#sim-metrics");
    if (metrics) metrics.innerHTML = metricsHtml();

    const table = host.querySelector("#sim-chunk");
    if (table) {
      table.innerHTML =
        "<thead><tr><th>rank</th><th>本步发送</th><th>→</th></tr></thead><tbody>" +
        info.rows.map((r) =>
          '<tr class="' + (sim.op === "sendrecv" && r.r !== 0 ? "" : "on") + '">' +
          "<td>" + r.r + "</td><td>" + esc(r.send) + "</td><td>" + esc(r.to) + "</td></tr>"
        ).join("") + "</tbody>";
    }
  }

  function metricsHtml() {
    const n = sim.n;
    const S = sim.sizeMb * 2 ** 20;
    const bw = (sim.bwGbps / 8) * 1e9; // bytes/s
    const lat = sim.latencyUs / 1e6;
    const steps = totalSteps();
    let perRank, total, secs, extra = "";

    if (sim.op === "sendrecv") {
      perRank = S;
      total = S;
      secs = lat + S / bw;
    } else {
      const ratio = sim.op === "allreduce" ? 2 * (n - 1) / n : (n - 1) / n;
      perRank = S * ratio;
      total = perRank * n;
      secs = steps * (lat + S / n / bw);
      if (sim.op === "allreduce") {
        const treeSteps = 2 * Math.ceil(Math.log2(n));
        const treeSecs = treeSteps * (lat + S / bw);
        extra = row("Tree 估算", time(treeSecs), treeSteps + " 步，小消息更低延迟");
      }
    }
    const busbw = secs > 0 ? perRank / secs : 0;
    return row("步数", steps + " 步", "每步一个 chunk/一跳") +
      row("本步每卡", bytes(S / (sim.op === "sendrecv" ? 1 : n)), "chunk = S / N") +
      row("每卡总发出", bytes(perRank), "所有步加起来") +
      row("网络总流量", bytes(total), "每卡 × N") +
      row("Ring 估算时间", time(secs), "延迟 + chunk/BW") +
      row("Bus bandwidth", (busbw / 1e9).toFixed(2) + " GB/s", "有效吞吐") +
      extra;
  }

  function ringSvg(n, op, step) {
    const cx = 160, cy = 150, R = 104;
    const pts = [];
    for (let i = 0; i < n; i++) {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / n;
      pts.push({ x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) });
    }
    let edges = "";
    for (let i = 0; i < n; i++) {
      const p = pts[i], q = pts[(i + 1) % n];
      const dx = q.x - p.x, dy = q.y - p.y;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len;
      const on = op !== "sendrecv" || i === 0;
      edges += '<line class="sim-edge' + (on ? " on" : "") + '" x1="' + (p.x + ux * 18) +
        '" y1="' + (p.y + uy * 18) + '" x2="' + (q.x - ux * 24) + '" y2="' + (q.y - uy * 24) +
        '" marker-end="url(#sim-arrow)"></line>';
    }
    const nodes = pts.map((p, i) =>
      '<g class="sim-node' + (op === "sendrecv" && i > 1 ? " dim" : "") + '">' +
      '<circle cx="' + p.x + '" cy="' + p.y + '" r="17"></circle>' +
      '<text x="' + p.x + '" y="' + (p.y + 4) + '" text-anchor="middle">' + i + "</text></g>"
    ).join("");
    return '<svg viewBox="0 0 320 300" role="img" aria-label="' + n + " ranks ring, step " + (step + 1) + '">' +
      '<defs><marker id="sim-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">' +
      '<path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"></path></marker></defs>' +
      '<g class="sim-edges">' + edges + "</g>" + nodes + "</svg>";
  }

  function row(label, value, sub) {
    return '<div class="sim-metric"><span>' + esc(label) + "</span><b>" + value +
      (sub ? "<em>" + esc(sub) + "</em>" : "") + "</b></div>";
  }

  function bytes(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + " MB";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + " KB";
    return Math.round(n) + " B";
  }

  function time(s) {
    if (s >= 1) return s.toFixed(2) + " s";
    if (s >= 1e-3) return (s * 1e3).toFixed(2) + " ms";
    return (s * 1e6).toFixed(1) + " µs";
  }

  /* ---------------------------------------------------------------- pd */

  function mountPd(host, data, world) {
    if (!host) return;
    const presets = world.presets;
    const gpus = world.gpus;
    const preset = presets.find((p) => p.id === pd.presetId) || presets[0];
    const gpu = gpus.find((g) => g.id === pd.gpu) || gpus[0];
    const kvScale = pd.kv === "fp8" ? 0.5 : 1;
    const kvB = preset.math.kv_bytes * kvScale * pd.prompt;
    const bw = (pd.bwGbps / 8) * 1e9;
    const transferMs = pd.latencyUs / 1000 + (kvB / bw) * 1000;
    const gemmFlops = 2 * preset.math.active * pd.prompt;
    const attnFlops = 4 * preset.layers * preset.q_heads * preset.head_dim * pd.prompt * pd.prompt;
    const prefillFlops = gemmFlops + attnFlops;
    const prefillDenom = gpu.bf16 * 1e12 * pd.prefillGpus;
    const gemmMs = (gemmFlops / prefillDenom) * 1000;
    const attnMs = (attnFlops / prefillDenom) * 1000;
    const prefillMs = gemmMs + attnMs;
    const decodeBytes = preset.math.active * 2 + kvB;
    const tpotMs = (decodeBytes / (gpu.bandwidth * 1e9 * pd.decodeGpus)) * 1000;
    const ttftMs = prefillMs + transferMs + tpotMs;
    const decodeTotalMs = tpotMs * pd.output;
    const colocatedMs = prefillMs + tpotMs;
    const linkReq = kvB > 0 ? bw / kvB : 0;
    const totalMs = Math.max(0.001, prefillMs + transferMs + decodeTotalMs);
    const pct = (x) => (x / totalMs) * 100;

    host.innerHTML =
      '<div class="calc-layout">' +
      '<div class="calc-controls">' +
      '<label class="calc-field"><span>LLM</span><select data-pd="preset">' +
      presets.map((p) => option(p.id, p.name, pd.presetId)).join("") + "</select></label>" +
      '<label class="calc-field"><span>Prompt <b>' + fmt(pd.prompt) + " tok</b></span>" +
      '<input type="range" data-pd="prompt" min="512" max="32768" step="512" value="' + pd.prompt + '"></label>' +
      '<label class="calc-field"><span>输出 <b>' + pd.output + " tok</b></span>" +
      '<input type="range" data-pd="output" min="16" max="512" step="16" value="' + pd.output + '"></label>' +
      '<label class="calc-field"><span>KV 精度</span><select data-pd="kv">' +
      option("fp16", "FP16（2 B）", pd.kv) + option("fp8", "FP8（1 B）", pd.kv) + "</select></label>" +
      '<label class="calc-field"><span>网络带宽</span><select data-pd="bw">' +
      [100, 200, 400, 800].map((v) => option(v, v + " Gb/s（" + v / 8 + " GB/s）", pd.bwGbps)).join("") + "</select></label>" +
      '<label class="calc-field"><span>单跳延迟</span><select data-pd="lat">' +
      [2, 5, 10, 20].map((v) => option(v, v + " µs", pd.latencyUs)).join("") + "</select></label>" +
      '<div class="train-par-row"><label class="calc-field"><span>prefill 卡</span><select data-pd="pg">' +
      [1, 2, 4, 8].map((v) => option(v, v, pd.prefillGpus)).join("") + "</select></label>" +
      '<label class="calc-field"><span>decode 卡</span><select data-pd="dg">' +
      [1, 2, 4, 8].map((v) => option(v, v, pd.decodeGpus)).join("") + "</select></label></div>" +
      '<div class="calc-field"><span>GPU</span><div class="gpu-row">' +
      gpus.map((g) => '<button class="gpu' + (g.id === gpu.id ? " on" : "") + '" data-pd-gpu="' + g.id + '">' + esc(g.name) + "</button>").join("") +
      "</div></div>" +
      "</div>" +

      '<div class="calc-result">' +
      '<div class="compare">' +
      '<div class="stat"><div class="stat-label">KV / 请求</div><div class="stat-value">' + bytes(kvB) +
      '</div><div class="stat-sub">' + fmt(pd.prompt) + " tok × " + bytes(preset.math.kv_bytes * kvScale) + "/token</div></div>" +
      '<div class="stat"><div class="stat-label">KV 传输</div><div class="stat-value">' + transferMs.toFixed(1) + " ms" +
      '</div><div class="stat-sub">' + pct(transferMs).toFixed(0) + "% of TTFT；走 RDMA 才有这个数</div></div>" +
      '<div class="stat"><div class="stat-label">TTFT</div><div class="stat-value">' + ttftMs.toFixed(0) + " ms" +
      '</div><div class="stat-sub">prefill ' + prefillMs.toFixed(0) + " + transfer " + transferMs.toFixed(0) + " + 首步 " + tpotMs.toFixed(1) + "</div></div>" +
      '<div class="stat"><div class="stat-label">TPOT</div><div class="stat-value">' + tpotMs.toFixed(1) + " ms" +
      '</div><div class="stat-sub">每 token 读激活权重 + 本请求 KV</div></div>' +
      '<div class="stat"><div class="stat-label">生成总时长</div><div class="stat-value">' + (totalMs / 1000).toFixed(2) + " s" +
      '</div><div class="stat-sub">prefill + transfer + ' + pd.output + " × TPOT</div></div>" +
      "</div>" +

      '<div class="pd-timeline"><div class="pd-seg pre" style="width:' + Math.max(1, pct(prefillMs)) + '%"><span>prefill</span></div>' +
      '<div class="pd-seg net" style="width:' + Math.max(1, pct(transferMs)) + '%"><span>KV transfer</span></div>' +
      '<div class="pd-seg dec" style="width:' + Math.max(1, pct(decodeTotalMs)) + '%"><span>decode × ' + pd.output + "</span></div></div>" +

      '<div class="pd-formula"><b>账本</b><p>' +
      "KV = " + bytes(preset.math.kv_bytes * kvScale) + "/tok × " + fmt(pd.prompt) + " = " + bytes(kvB) + "；" +
      "transfer = " + pd.latencyUs + " µs + " + bytes(kvB) + " / " + (pd.bwGbps / 8) + " GB/s = " + transferMs.toFixed(1) + " ms；" +
      "prefill = GEMM（2 × active × prompt）" + gemmMs.toFixed(0) + " ms + attention（4 × layers × q_heads × head_dim × prompt²）" +
      attnMs.toFixed(0) + " ms = " + prefillMs.toFixed(0) + " ms；" +
      "TPOT ≈ (激活权重 + KV) / (" + gpu.bandwidth + " GB/s × " + pd.decodeGpus + ") = " + tpotMs.toFixed(1) + " ms。" +
      "</p></div>" +

      '<p class="note">同机不迁移时 TTFT ≈ ' + colocatedMs.toFixed(0) + " ms；PD 额外付出 " + transferMs.toFixed(1) +
      " ms 的 KV 传输，换来 prefill/decode 独立扩缩容。输出 " + pd.output + " tok 的 decode 约 " +
      (decodeTotalMs / 1000).toFixed(2) + " s。链路最多支撑约 <b>" + linkReq.toFixed(0) +
      "</b> 个请求/s 的 KV 传输（只算网络，不含 compute 和排队）。" +
      (transferMs > prefillMs
        ? " 当前 KV 传输比 prefill 还慢：先换 RDMA / 提高带宽，或减少 prompt/走 prefix cache。"
        : " 当前 prefill 仍是 TTFT 大头；网络不是单请求瓶颈，重点看并发下的容量。") +
      "</p>" +
      '<p class="note">PD 的收益在吞吐与资源隔离：prefill 池可以按 prompt 长度扩，decode 池按输出长度/并发扩；' +
      "如果 prompt 很短、输出也很短，迁移开销可能盖过收益。</p>" +
      "</div></div>";

    host.querySelectorAll("[data-pd]").forEach((inp) => {
      inp.addEventListener("change", () => {
        const f = inp.dataset.pd;
        if (f === "preset" || f === "kv") pd[f === "preset" ? "presetId" : "kv"] = inp.value;
        else if (["prompt", "output", "bw", "lat", "pg", "dg"].includes(f)) {
          if (f === "bw") pd.bwGbps = +inp.value;
          else if (f === "lat") pd.latencyUs = +inp.value;
          else if (f === "pg") pd.prefillGpus = +inp.value;
          else if (f === "dg") pd.decodeGpus = +inp.value;
          else pd[f] = +inp.value;
        }
        mountPd(host, data, world);
      });
    });
    host.querySelectorAll("[data-pd-gpu]").forEach((b) =>
      b.addEventListener("click", () => {
        pd.gpu = b.dataset.pdGpu;
        mountPd(host, data, world);
      })
    );
  }

  N.stop = stopSim;

  /* ------------------------------------------------------------- helpers */

  function section(sec, body) {
    return '<h2 class="sec-title">' + esc(sec.title) + "</h2>" +
      '<p class="page-lead">' + esc(sec.lead) + "</p>" + body();
  }

  function control(label, input) {
    return '<label class="sim-field"><span>' + esc(label) + "</span>" + input + "</label>";
  }

  function option(value, label, current) {
    return '<option value="' + esc(value) + '"' + (String(value) === String(current) ? " selected" : "") +
      ">" + esc(label) + "</option>";
  }

  function details(q, a) {
    return '<details class="thinkbox"><summary>' + esc(q) + "</summary><p>" + esc(a) + "</p></details>";
  }

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
})();
