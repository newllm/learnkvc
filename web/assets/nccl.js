/* NCCL deep dive: collectives, ring/tree players, alpha-beta scan, topology ledger.
 * Content lives in web/tools/nccl.json; example/nccl_sim.py supplies simCode
 * and simOutput through build.py. Classic script, file:// safe. */
(function () {
  const LK = (window.LK = window.LK || {});
  const N = (LK.nccl = LK.nccl || {});

  const KIB = 1 << 10;
  const MIB = 1 << 20;
  const GIB = 1 << 30;
  const BW_BYTES = 50e9; // 400 Gb/s = 50 GB/s, same as nccl_sim.py
  const LAT_S = 2e-6; // 2 us per hop

  const SIZES_MIB = [1, 16, 128, 512];

  const play = { algo: "ring", n: 8, sizeMiB: 128, step: 0, playing: false, timer: null };
  const scan = { n: 8, bwGbps: 400, latencyUs: 2 };
  const topo = { link: "ib400", n: 8, sizeIdx: 3 };

  const LINKS = [
    { id: "nvlink", name: "NVLink 4（同机）", gbs: 450, latUs: 0.5, note: "单机 GPU 全互联，TP 首选" },
    { id: "nvswitch", name: "NVSwitch + NVLS", gbs: 900, latUs: 0.4, note: "单机全互联，交换机内归约" },
    { id: "pcie4", name: "PCIe 4.0 x16", gbs: 16, latUs: 1.2, note: "无 NVLink 的单机" },
    { id: "pcie5", name: "PCIe 5.0 x16", gbs: 32, latUs: 1.0, note: "新平台单机" },
    { id: "ib200", name: "InfiniBand HDR 200G", gbs: 25, latUs: 1.5, note: "跨机，支持 SHARP" },
    { id: "ib400", name: "InfiniBand NDR 400G", gbs: 50, latUs: 2.0, note: "跨机主流，本章 LLM" },
    { id: "roce", name: "RoCEv2 400G", gbs: 50, latUs: 3.0, note: "跨机以太网，需 PFC + ECN" },
    { id: "tcp", name: "TCP Socket", gbs: 6, latUs: 80, note: "兜底，慢一个量级" }
  ];

  const TOPO_SIZES = [
    { label: "4 KiB（decode TP）", bytes: 4 * KIB },
    { label: "64 KiB", bytes: 64 * KIB },
    { label: "1 MiB", bytes: 1 * MIB },
    { label: "8 MiB（prefill TP）", bytes: 8 * MIB },
    { label: "128 MiB", bytes: 128 * MIB }
  ];

  /* ------------------------------------------------------------- render */

  N.render = function (el, data, world) {
    stopPlay();
    if (!el || !data) return;
    const d = data;
    el.innerHTML =
      '<p class="kicker">' + esc(d.kicker) + "</p>" +
      '<h1 class="page-title">' + esc(d.title) + "</h1>" +
      '<p class="page-lead">' + esc(d.lead) + "</p>" +
      whyHtml(d.why) +
      collectivesHtml(d.collectives) +
      ringHtml(d.ring) +
      treeHtml(d.tree) +
      modelHtml(d.model) +
      topologyHtml(d.topology) +
      measureHtml(d.measure) +
      codeHtml(d.codeSection, d.simOutput) +
      checkHtml(d.check);

    mountPlayer(el.querySelector("[data-nccl-player]"));
    mountScan(el.querySelector("[data-nccl-scan]"));
    mountTopo(el.querySelector("[data-nccl-topo]"));

    const codeEl = el.querySelector("#nccl-sim-code");
    if (codeEl && LK.codeview && d.simCode && d.simCode.lines) LK.codeview.mount(codeEl, d.simCode.lines);
  };

  N.stop = function () {
    stopPlay();
  };

  /* -------------------------------------------------------------- why */

  function whyHtml(sec) {
    if (!sec) return "";
    return section(
      sec,
      '<div class="net-cards">' +
        (sec.contrast || [])
          .map(
            (c) =>
              '<div class="net-card"><b>' + esc(c.name) + "</b>" +
              '<p><em class="nccl-dim">' + esc(c.scope) + "</em> " + esc(c.answers) + "</p></div>"
          )
          .join("") +
        "</div>" +
        '<div class="table-wrap"><table class="tagtable">' +
        "<thead><tr><th>并行方式</th><th>落到哪个 collective</th><th>消息大小</th><th>调度 / 网络含义</th></tr></thead><tbody>" +
        (sec.mapping || [])
          .map(
            (m) =>
              "<tr><td><b>" + esc(m.name) + "</b></td><td>" + esc(m.collective) +
              "</td><td>" + esc(m.size) + "</td><td>" + esc(m.when) + "</td></tr>"
          )
          .join("") +
        "</tbody></table></div>" +
        (sec.decode ? '<p class="note">' + esc(sec.decode) + "</p>" : "") +
        (sec.kv ? '<p class="note">' + esc(sec.kv) + "</p>" : "") +
        takeaway(sec)
    );
  }

  /* ------------------------------------------------------ collectives */

  function collectivesHtml(sec) {
    if (!sec) return "";
    return section(
      sec,
      '<div class="table-wrap"><table class="tagtable nccl-collective-table">' +
        "<thead><tr><th>Collective</th><th>NCCL API</th><th>语义</th><th>每 rank 字节</th><th>步数</th><th>推理里的用途</th></tr></thead><tbody>" +
        (sec.rows || [])
          .map(
            (r) =>
              "<tr><td><b>" + esc(r.name) + "</b></td><td><code>" + esc(r.api) + "</code></td><td>" +
              esc(r.what) + "</td><td>" + esc(r.bytes) + "</td><td>" + esc(r.steps) +
              "</td><td>" + esc(r.use) + "</td></tr>"
          )
          .join("") +
        "</tbody></table></div>" +
        (sec.rules ? '<ul class="train-rules">' + sec.rules.map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul>" : "") +
        (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "") +
        takeaway(sec)
    );
  }

  /* ------------------------------------------------------------- ring */

  function ringHtml(sec) {
    if (!sec) return "";
    return section(
      sec,
      '<div class="pd-flow">' +
        (sec.stages || [])
          .map(
            (s, i) =>
              '<div class="pd-step"><span class="train-no">0' + (i + 1) + "</span><span><b>" +
              esc(s.name) + "</b> " + esc(s.detail) + "</span></div>"
          )
          .join("") +
        "</div>" +
        '<div class="nccl-player" data-nccl-player></div>' +
        (sec.rules ? '<ul class="train-rules">' + sec.rules.map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul>" : "") +
        (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "") +
        takeaway(sec)
    );
  }

  /* ------------------------------------------------------------- tree */

  function treeHtml(sec) {
    if (!sec) return "";
    return section(
      sec,
      '<div class="table-wrap"><table class="tagtable">' +
        "<thead><tr><th>算法</th><th>步数</th><th>每步发送</th><th>带宽表现</th><th>适合的消息</th></tr></thead><tbody>" +
        (sec.compare || [])
          .map(
            (c) =>
              "<tr><td><b>" + esc(c.name) + "</b></td><td>" + esc(c.steps) + "</td><td>" +
              esc(c.bytes) + "</td><td>" + esc(c.bandwidth) + "</td><td>" + esc(c.when) + "</td></tr>"
          )
          .join("") +
        "</tbody></table></div>" +
        '<h3 class="sec-sub">N=8 的 tree 调度：上行 reduce + 下行 broadcast</h3>' +
        '<div class="table-wrap"><table class="tagtable">' +
        "<thead><tr><th>step</th><th>phase</th><th>level</th><th>senders → receivers</th></tr></thead><tbody>" +
        (sec.walkthrough || [])
          .map(
            (w) =>
              "<tr><td>" + esc(w.step) + "</td><td>" + esc(w.phase) + "</td><td>" + esc(w.level) +
              '</td><td><code>' + esc(w.pairs) + "</code></td></tr>"
          )
          .join("") +
        "</tbody></table></div>" +
        (sec.rules ? '<ul class="train-rules">' + sec.rules.map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul>" : "") +
        (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "") +
        takeaway(sec)
    );
  }

  /* ------------------------------------------------------------ model */

  function modelHtml(sec) {
    if (!sec) return "";
    return section(
      sec,
      '<div class="train-ledger">' +
        (sec.formulas || [])
          .map(
            (f) =>
              '<div class="train-ledger-card"><b>' + esc(f.name) + "</b>" +
              '<code class="nccl-formula">' + esc(f.expr) + "</code>" +
              "<p>" + esc(f.detail) + "</p></div>"
          )
          .join("") +
        "</div>" +
        '<div class="nccl-scan" data-nccl-scan></div>' +
        '<h3 class="sec-sub">消息大小扫描（和 nccl_sim.py 同一组参数）</h3>' +
        '<div class="table-wrap"><table class="tagtable">' +
        "<thead><tr><th>消息大小 S</th><th>ring 耗时</th><th>tree 耗时</th><th>谁更快</th></tr></thead><tbody>" +
        (sec.sweep || [])
          .map(
            (s) =>
              "<tr><td><b>" + esc(s.size) + "</b></td><td>" + esc(s.ring) + "</td><td>" +
              esc(s.tree) + '</td><td class="' + (s.winner === "tree" ? "good" : "ok") + '">' +
              esc(s.winner) + "</td></tr>"
          )
          .join("") +
        "</tbody></table></div>" +
        (sec.crossover
          ? '<p class="note"><b>' + esc(sec.crossover.expr) + "</b>：" + esc(sec.crossover.why) + "</p>"
          : "") +
        '<h3 class="sec-sub">协议与交换机内归约</h3>' +
        '<div class="table-wrap"><table class="tagtable">' +
        "<thead><tr><th>协议 / 机制</th><th>什么时候用</th><th>细节</th></tr></thead><tbody>" +
        (sec.protocols || [])
          .map(
            (p) =>
              "<tr><td><b>" + esc(p.name) + "</b></td><td>" + esc(p.when) + "</td><td>" +
              esc(p.detail) + "</td></tr>"
          )
          .join("") +
        "</tbody></table></div>" +
        (sec.extras ? '<ul class="train-rules">' + sec.extras.map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul>" : "") +
        (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "") +
        takeaway(sec)
    );
  }

  /* --------------------------------------------------------- topology */

  function topologyHtml(sec) {
    if (!sec) return "";
    return section(
      sec,
      '<div class="table-wrap"><table class="tagtable">' +
        "<thead><tr><th>链路</th><th>带宽（每卡单向）</th><th>延迟</th><th>范围 / 备注</th></tr></thead><tbody>" +
        (sec.rows || [])
          .map(
            (r) =>
              "<tr><td><b>" + esc(r.link) + "</b></td><td>" + esc(r.bw) + "</td><td>" +
              esc(r.latency) + "</td><td>" + esc(r.scope) + "</td></tr>"
          )
          .join("") +
        "</tbody></table></div>" +
        '<h3 class="sec-sub">怎么读 nvidia-smi topo -m</h3>' +
        (sec.reading
          ? '<ul class="train-rules">' + sec.reading.map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul>"
          : "") +
        (sec.rules ? '<ul class="train-rules">' + sec.rules.map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul>" : "") +
        '<div class="nccl-topo" data-nccl-topo></div>' +
        (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "") +
        takeaway(sec)
    );
  }

  /* ---------------------------------------------------------- measure */

  function measureHtml(sec) {
    if (!sec) return "";
    return section(
      sec,
      '<h3 class="sec-sub">可以直接跑的命令</h3>' +
        (sec.commands
          ? '<pre class="train-run"><code>' + esc(sec.commands.join("\n")) + "</code></pre>"
          : "") +
        '<div class="table-wrap"><table class="tagtable">' +
        "<thead><tr><th>变量</th><th>示例</th><th>作用 / 排查点</th></tr></thead><tbody>" +
        (sec.envs || [])
          .map(
            (e) =>
              "<tr><td><code>" + esc(e.name) + "</code></td><td><code>" + esc(e.value) +
              "</code></td><td>" + esc(e.why) + "</td></tr>"
          )
          .join("") +
        "</tbody></table></div>" +
        (sec.checklist
          ? '<h3 class="sec-sub">排查清单</h3><ul class="train-rules train-checklist">' +
            sec.checklist.map((c) => "<li>" + esc(c) + "</li>").join("") +
            "</ul>"
          : "") +
        (sec.pitfalls
          ? '<div class="fncaveats"><b>常见坑</b><ul>' +
            sec.pitfalls.map((p) => "<li>" + esc(p) + "</li>").join("") +
            "</ul></div>"
          : "") +
        (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "") +
        takeaway(sec)
    );
  }

  /* ------------------------------------------------------------- code */

  function codeHtml(sec, simOutput) {
    if (!sec) return "";
    return section(
      sec,
      '<pre class="train-run"><code>' + esc(sec.run || "") + "</code></pre>" +
        (sec.notes ? '<ul class="train-rules">' + sec.notes.map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul>" : "") +
        '<div class="train-code-head"><span>' + esc(sec.file || "example/nccl_sim.py") +
        "</span><span>纯 Python · 可运行</span></div>" +
        '<div class="train-code" id="nccl-sim-code"></div>' +
        '<h3 class="sec-sub">示例输出</h3>' +
        '<pre class="train-run"><code>' + esc(simOutput || "") + "</code></pre>" +
        takeaway(sec)
    );
  }

  /* ------------------------------------------------------------ check */

  function checkHtml(sec) {
    if (!sec) return "";
    return section(
      sec,
      (sec.checklist
        ? '<ul class="train-rules train-checklist">' +
          sec.checklist.map((c) => "<li>" + esc(c) + "</li>").join("") +
          "</ul>"
        : "") +
        (sec.questions || []).map((q) => details(q.q, q.a)).join("") +
        takeaway(sec)
    );
  }

  /* ===================================================== ring/tree player */

  function mountPlayer(host) {
    if (!host) return;
    host.innerHTML =
      '<div class="sim-controls">' +
      control("算法", '<select data-nccl-pl="algo">' +
        option("ring", "Ring", play.algo) + option("tree", "Tree", play.algo) + "</select>") +
      control("Rank 数", '<select data-nccl-pl="n">' +
        [2, 4, 8].map((v) => option(v, v + " 卡", play.n)).join("") + "</select>") +
      control("消息大小 S", '<select data-nccl-pl="size">' +
        SIZES_MIB.map((v) => option(v, v + " MiB", play.sizeMiB)).join("") + "</select>") +
      "</div>" +
      '<div class="sim-body">' +
      '<div class="sim-ring" data-nccl-pl="viz"></div>' +
      '<div class="sim-side">' +
      '<div class="sim-phase" data-nccl-pl="phase"></div>' +
      '<div class="sim-metrics" data-nccl-pl="metrics"></div>' +
      "</div></div>" +
      '<div class="sim-stepbar">' +
      '<button class="nav-btn" type="button" data-nccl-pl-step="-1" aria-label="上一步">←</button>' +
      '<button class="btn" type="button" data-nccl-pl-play>▶ 播放</button>' +
      '<button class="nav-btn" type="button" data-nccl-pl-step="1" aria-label="下一步">→</button>' +
      '<span class="nccl-step-label" data-nccl-pl="label"></span>' +
      '<div class="sim-progress"><i data-nccl-pl="progress"></i></div>' +
      '<button class="btn ghost" type="button" data-nccl-pl-reset>重置</button>' +
      "</div>" +
      '<div class="sim-chunk-wrap"><table class="sim-chunk" data-nccl-pl="table"></table></div>' +
      '<p class="note nccl-hint">时间模型：每步 = 2 µs + 本步字节 / 50 GB/s，与 example/nccl_sim.py 同参数；播放到末尾会自动停止。</p>';

    host.querySelectorAll("select[data-nccl-pl]").forEach((inp) => {
      inp.addEventListener("change", () => {
        const f = inp.dataset.ncclPl;
        if (f === "algo") play.algo = inp.value;
        else if (f === "n") play.n = +inp.value;
        else if (f === "size") play.sizeMiB = +inp.value;
        stopPlay();
        play.step = 0;
        updatePlayer(host);
      });
    });
    host.querySelectorAll("[data-nccl-pl-step]").forEach((b) =>
      b.addEventListener("click", () => {
        stopPlay();
        play.step = clamp(play.step + +b.dataset.ncclPlStep, 0, totalSteps() - 1);
        updatePlayer(host);
      })
    );
    host.querySelector("[data-nccl-pl-play]").addEventListener("click", () => {
      if (play.playing) stopPlay();
      else startPlay(host);
      updatePlayer(host);
    });
    host.querySelector("[data-nccl-pl-reset]").addEventListener("click", () => {
      stopPlay();
      play.step = 0;
      updatePlayer(host);
    });
    updatePlayer(host);
  }

  function totalSteps() {
    return play.algo === "ring" ? 2 * (play.n - 1) : 2 * treeLevels(play.n);
  }

  function treeLevels(n) {
    return Math.max(1, Math.round(Math.log2(n)));
  }

  function startPlay(host) {
    if (play.step >= totalSteps() - 1) play.step = 0;
    play.playing = true;
    play.timer = setInterval(() => {
      if (play.step >= totalSteps() - 1) {
        stopPlay();
        updatePlayer(host);
        return;
      }
      play.step += 1;
      updatePlayer(host);
    }, 700);
  }

  function stopPlay() {
    play.playing = false;
    if (play.timer) clearInterval(play.timer);
    play.timer = null;
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function playerInfo() {
    const n = play.n;
    const size = play.sizeMiB * MIB;
    const step = play.step;
    if (play.algo === "ring") {
      const rs = n - 1;
      const phase = step < rs ? "reduce-scatter" : "all-gather";
      const local = step < rs ? step : step - rs;
      const chunk = size / n;
      const rows = [];
      for (let r = 0; r < n; r++) {
        const c = phase === "reduce-scatter" ? (r - local + n) % n : (r - local + 1 + n) % n;
        rows.push({
          rank: r,
          active: true,
          action: "chunk c" + c,
          dest: "→ rank " + ((r + 1) % n),
          bytes: chunk,
          cum: (step + 1) * chunk,
        });
      }
      const perRankTotal = (size * 2 * (n - 1)) / n;
      const secs = 2 * (n - 1) * (LAT_S + chunk / BW_BYTES);
      return {
        phase,
        stepName:
          phase + " 第 " + (local + 1) + " / " + rs + " 步：每个 rank 把 chunk (r − " + local +
          ") mod " + n + " 发给右邻居；每卡本步 " + fmtBytes(chunk) + "，累计 " + fmtBytes((step + 1) * chunk) + "。",
        rows,
        viz: { kind: "ring", n, step },
        metrics: {
          steps: totalSteps(),
          stepBytes: fmtBytes(chunk),
          stepTime: fmtTime(LAT_S + chunk / BW_BYTES),
          cumBytes: fmtBytes((step + 1) * chunk),
          cumTime: fmtTime((step + 1) * (LAT_S + chunk / BW_BYTES)),
          perRank: fmtBytes(perRankTotal),
          busbw: (perRankTotal / secs / 1e9).toFixed(2) + " GB/s",
          network: fmtBytes(perRankTotal * n),
        },
      };
    }

    const L = treeLevels(n);
    const reduce = step < L;
    const phase = reduce ? "reduce" : "broadcast";
    const k = reduce ? step + 1 : step - L + 1;
    const sends = treeSends(n, reduce, k);
    const sentByRank = {};
    sends.forEach((s) => {
      sentByRank[s.from] = s.to;
    });
    const receivers = {};
    sends.forEach((s) => {
      receivers[s.to] = true;
    });
    // per-rank bytes sent up to and including this step
    const cumByRank = [];
    for (let r = 0; r < n; r++) cumByRank[r] = 0;
    for (let s = 0; s <= step; s++) {
      const isRed = s < L;
      const kk = isRed ? s + 1 : s - L + 1;
      treeSends(n, isRed, kk).forEach((x) => {
        cumByRank[x.from] += size;
      });
    }
    const rows = [];
    for (let r = 0; r < n; r++) {
      const active = Object.prototype.hasOwnProperty.call(sentByRank, r);
      rows.push({
        rank: r,
        active,
        action: active ? "发整块 S" : receivers[r] ? "接收" : "—",
        dest: active ? "→ rank " + sentByRank[r] : "",
        bytes: active ? size : 0,
        cum: cumByRank[r],
      });
    }
    const perRankAvg = (size * 2 * (n - 1)) / n;
    const secs = 2 * L * (LAT_S + size / BW_BYTES);
    const stepBytes = size;
    return {
      phase,
      stepName:
        phase + " 第 " + k + " / " + L + " 层：" + sends.map((s) => "rank " + s.from + " → " + s.to).join("、") +
        "；每步搬整块 S = " + fmtBytes(size) + "。",
      rows,
      viz: { kind: "tree", n, step },
      metrics: {
        steps: totalSteps(),
        stepBytes: fmtBytes(stepBytes) + "（发送者）",
        stepTime: fmtTime(LAT_S + stepBytes / BW_BYTES),
        cumBytes: fmtBytes(Math.max.apply(null, cumByRank)),
        cumTime: fmtTime((step + 1) * (LAT_S + stepBytes / BW_BYTES)),
        perRank: fmtBytes(perRankAvg) + "（平均）",
        busbw: (perRankAvg / secs / 1e9).toFixed(2) + " GB/s",
        network: fmtBytes(perRankAvg * n),
      },
    };
  }

  function treeSends(n, reduce, k) {
    const L = treeLevels(n);
    const sends = [];
    if (reduce) {
      const half = Math.pow(2, k - 1);
      for (let r = 0; r < n; r++) if (r % Math.pow(2, k) === half) sends.push({ from: r, to: r - half });
    } else {
      const span = Math.pow(2, L - k);
      const mod = Math.pow(2, L - k + 1);
      for (let r = 0; r < n; r++) if (r % mod === 0) sends.push({ from: r, to: r + span });
    }
    return sends;
  }

  function updatePlayer(host) {
    if (!host) return;
    const info = playerInfo();
    const label = host.querySelector('[data-nccl-pl="label"]');
    if (label) label.textContent = "step " + (play.step + 1) + " / " + totalSteps() + " · " + info.phase;
    const prog = host.querySelector('[data-nccl-pl="progress"]');
    if (prog) prog.style.width = (((play.step + 1) / totalSteps()) * 100).toFixed(1) + "%";
    const playBtn = host.querySelector("[data-nccl-pl-play]");
    if (playBtn) {
      playBtn.textContent =
        play.playing ? "⏸ 暂停" : play.step >= totalSteps() - 1 ? "↻ 重播" : "▶ 播放";
    }
    const phase = host.querySelector('[data-nccl-pl="phase"]');
    if (phase) {
      phase.innerHTML =
        "<b>" + esc(play.algo === "ring" ? "Ring all-reduce" : "Tree all-reduce") + "</b>" +
        "<span>" + esc(info.stepName) + "</span>";
    }
    const viz = host.querySelector('[data-nccl-pl="viz"]');
    if (viz) viz.innerHTML = info.viz.kind === "ring" ? ringSvg(info.viz.n, play.step) : treeSvg(info.viz.n, play.step);
    const metrics = host.querySelector('[data-nccl-pl="metrics"]');
    if (metrics) {
      metrics.innerHTML =
        row("步数", info.metrics.steps + " 步", info.phase) +
        row("本步发送", info.metrics.stepBytes, "每步一个 chunk / 整块") +
        row("本步耗时", info.metrics.stepTime, "α + bytes/BW") +
        row("累计发出", info.metrics.cumBytes, "到本步为止（每卡）") +
        row("累计时间", info.metrics.cumTime, "到本步为止") +
        row("每卡总发出", info.metrics.perRank, "走完全部步数") +
        row("busbw（全程）", info.metrics.busbw, "每卡字节 / 总时间") +
        row("网络总流量", info.metrics.network, "N × 每卡字节");
    }
    const table = host.querySelector('[data-nccl-pl="table"]');
    if (table) {
      table.innerHTML =
        "<thead><tr><th>rank</th><th>本步动作</th><th>去向</th><th>累计发出</th></tr></thead><tbody>" +
        info.rows
          .map(
            (r) =>
              '<tr class="' + (r.active ? "on" : "") + '"><td>' + r.rank + "</td><td>" +
              esc(r.action) + "</td><td>" + esc(r.dest || "—") + "</td><td>" + fmtBytes(r.cum) + "</td></tr>"
          )
          .join("") +
        "</tbody>";
    }
  }

  function ringSvg(n, step) {
    const cx = 160;
    const cy = 132;
    const R = 96;
    const pts = [];
    for (let i = 0; i < n; i++) {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / n;
      pts.push({ x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) });
    }
    let edges = "";
    for (let i = 0; i < n; i++) {
      const p = pts[i];
      const q = pts[(i + 1) % n];
      const dx = q.x - p.x;
      const dy = q.y - p.y;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len;
      const uy = dy / len;
      edges +=
        '<line class="sim-edge on" x1="' + (p.x + ux * 19) + '" y1="' + (p.y + uy * 19) +
        '" x2="' + (q.x - ux * 22) + '" y2="' + (q.y - uy * 22) + '"></line>';
    }
    const nodes = pts
      .map(
        (p, i) =>
          '<g class="sim-node"><circle cx="' + p.x + '" cy="' + p.y + '" r="18"></circle>' +
          '<text x="' + p.x + '" y="' + (p.y + 4) + '" text-anchor="middle">' + i + "</text></g>"
      )
      .join("");
    return (
      '<svg viewBox="0 0 320 264" role="img" aria-label="' + n + " ranks ring, step " + (step + 1) +
      '"><g class="sim-edges">' + edges + "</g>" + nodes + "</svg>"
    );
  }

  function treeSvg(n, step) {
    const L = treeLevels(n);
    const W = 320;
    const H = 224;
    const margin = 24;
    const leafY = 196;
    const rootY = 26;
    const spanX = W - margin * 2;
    const leafX = (i) => margin + (i * spanX) / (n - 1);
    const yOf = (level) => leafY - (level * (leafY - rootY)) / L;
    const pos = (level, idx) => {
      const count = n / Math.pow(2, level);
      const leftLeaf = idx * Math.pow(2, level);
      const rightLeaf = leftLeaf + Math.pow(2, level) - 1;
      return { x: (leafX(leftLeaf) + leafX(rightLeaf)) / 2, y: yOf(level) };
    };
    const active = {};
    if (step < L) {
      const k = step + 1;
      treeSends(n, true, k).forEach((s) => {
        const a = { level: k - 1, idx: s.from >> (k - 1) };
        const b = { level: k, idx: s.from >> k };
        active[a.level + ":" + a.idx + "-" + b.level + ":" + b.idx] = true;
      });
    } else {
      const k = step - L + 1;
      treeSends(n, false, k).forEach((s) => {
        const hi = { level: L - k + 1, idx: s.from >> (L - k + 1) };
        const lo = { level: L - k, idx: s.to >> (L - k) };
        active[lo.level + ":" + lo.idx + "-" + hi.level + ":" + hi.idx] = true;
      });
    }
    let edges = "";
    for (let level = 0; level < L; level++) {
      const count = n / Math.pow(2, level);
      for (let idx = 0; idx < count; idx++) {
        const p = pos(level, idx);
        const q = pos(level + 1, idx >> 1);
        const on = active[level + ":" + idx + "-" + (level + 1) + ":" + (idx >> 1)];
        edges +=
          '<line class="nccl-tree-edge' + (on ? " on" : "") + '" x1="' + p.x + '" y1="' + p.y +
          '" x2="' + q.x + '" y2="' + q.y + '"></line>';
      }
    }
    let hubs = "";
    for (let level = 1; level <= L; level++) {
      const count = n / Math.pow(2, level);
      for (let idx = 0; idx < count; idx++) {
        const p = pos(level, idx);
        hubs += '<circle class="nccl-tree-hub" cx="' + p.x + '" cy="' + p.y + '" r="7"></circle>';
      }
    }
    let leaves = "";
    for (let r = 0; r < n; r++) {
      const p = pos(0, r);
      leaves +=
        '<g class="sim-node"><circle cx="' + p.x + '" cy="' + p.y + '" r="16"></circle>' +
        '<text x="' + p.x + '" y="' + (p.y + 4) + '" text-anchor="middle">' + r + "</text></g>";
    }
    return (
      '<svg viewBox="0 0 320 224" role="img" aria-label="tree all-reduce, ' + n + " ranks, step " + (step + 1) +
      '"><g class="sim-edges">' + edges + "</g>" + hubs + leaves + "</svg>"
    );
  }

  /* ================================================== alpha-beta scan */

  function mountScan(host) {
    if (!host) return;
    host.innerHTML =
      '<div class="sim-controls nccl-scan-controls">' +
      control("Rank 数", '<select data-nccl-sc="n">' +
        [2, 4, 8, 16].map((v) => option(v, v + " 卡", scan.n)).join("") + "</select>") +
      control("链路带宽", '<select data-nccl-sc="bw">' +
        [100, 200, 400, 800].map((v) => option(v, v + " Gb/s（" + v / 8 + " GB/s）", scan.bwGbps)).join("") +
        "</select>") +
      control("单跳延迟 α", '<select data-nccl-sc="lat">' +
        [1, 2, 5, 10].map((v) => option(v, v + " µs", scan.latencyUs)).join("") + "</select>") +
      "</div>" +
      '<div class="nccl-charts">' +
      '<figure class="nccl-chart"><figcaption>耗时（对数纵轴，越低越好）</figcaption><div data-nccl-sc="timeChart"></div></figure>' +
      '<figure class="nccl-chart"><figcaption>busbw（GB/s，越高越好）</figcaption><div data-nccl-sc="bwChart"></div></figure>' +
      "</div>" +
      '<div class="nccl-legend"><span class="ring">Ring</span><span class="tree">Tree</span><span class="cross">crossover</span></div>' +
      '<div class="compare" data-nccl-sc="stats"></div>' +
      '<p class="note nccl-scan-formula" data-nccl-sc="formula"></p>';

    host.querySelectorAll("select[data-nccl-sc]").forEach((inp) => {
      inp.addEventListener("change", () => {
        const f = inp.dataset.ncclSc;
        if (f === "n") scan.n = +inp.value;
        else if (f === "bw") scan.bwGbps = +inp.value;
        else if (f === "lat") scan.latencyUs = +inp.value;
        updateScan(host);
      });
    });
    updateScan(host);
  }

  function scanParams() {
    return {
      n: scan.n,
      bw: (scan.bwGbps / 8) * 1e9,
      lat: scan.latencyUs / 1e6,
      levels: treeLevels(scan.n),
    };
  }

  function scanTimes(s, p) {
    const ring = 2 * (p.n - 1) * (p.lat + s / p.n / p.bw);
    const tree = 2 * p.levels * (p.lat + s / p.bw);
    return { ring, tree };
  }

  function scanCrossover(p) {
    const denom = p.levels - (p.n - 1) / p.n;
    if (denom <= 0) return 0;
    const v = (p.bw * p.lat * (p.n - 1 - p.levels)) / denom;
    return v > 0 ? v : 0;
  }

  function updateScan(host) {
    if (!host) return;
    const p = scanParams();
    const cross = scanCrossover(p);
    const sizes = [];
    for (let e = 10; e <= 30; e += 0.5) sizes.push(Math.pow(2, e));
    const timeChart = host.querySelector('[data-nccl-sc="timeChart"]');
    if (timeChart) timeChart.innerHTML = timeChartSvg(sizes, p, cross);
    const bwChart = host.querySelector('[data-nccl-sc="bwChart"]');
    if (bwChart) bwChart.innerHTML = busbwChartSvg(sizes, p, cross);

    const s4k = 4 * KIB;
    const s128m = 128 * MIB;
    const t4k = scanTimes(s4k, p);
    const t128 = scanTimes(s128m, p);
    const switchTime = cross > 0 ? scanTimes(cross, p).ring : 0;
    const stats = host.querySelector('[data-nccl-sc="stats"]');
    if (stats) {
      stats.innerHTML =
        '<div class="stat"><div class="stat-label">crossover S*</div><div class="stat-value">' +
        (cross > 0 ? fmtBytes(cross) : "无") +
        '</div><div class="stat-sub">' + (cross > 0 ? "小于它 tree 快，大于它 ring 快" : "N=2 时 ring 始终不慢于 tree") +
        "</div></div>" +
        '<div class="stat"><div class="stat-label">crossover 时耗时</div><div class="stat-value">' +
        (cross > 0 ? fmtTime(switchTime) : "—") +
        '</div><div class="stat-sub">此时 T_ring = T_tree</div></div>' +
        '<div class="stat"><div class="stat-label">4 KiB（decode）</div><div class="stat-value">' +
        (t4k.tree < t4k.ring ? "tree" : "ring") +
        '</div><div class="stat-sub">tree ' + fmtTime(t4k.tree) + " · ring " + fmtTime(t4k.ring) + "</div></div>" +
        '<div class="stat"><div class="stat-label">128 MiB（大消息）</div><div class="stat-value">' +
        (t128.ring < t128.tree ? "ring" : "tree") +
        '</div><div class="stat-sub">ring ' + fmtTime(t128.ring) + " · tree " + fmtTime(t128.tree) + "</div></div>";
    }
    const formula = host.querySelector('[data-nccl-sc="formula"]');
    if (formula) {
      formula.textContent =
        "当前：N=" + p.n + "，α=" + scan.latencyUs + " µs，BW=" + scan.bwGbps / 8 + " GB/s；" +
        "T_ring = 2(N−1)(α + S/(N·BW))，T_tree = 2⌈log₂N⌉(α + S/BW)";
      formula.textContent += cross > 0
        ? "；S* = BW·α·[(N−1)−⌈log₂N⌉] / [⌈log₂N⌉−(N−1)/N] ≈ " + fmtBytes(cross) + "。"
        : "；N=2 时 tree 没有优势区。";
    }
  }

  function timeChartSvg(sizes, p, cross) {
    const W = 340;
    const H = 190;
    const pad = { l: 62, r: 18, t: 8, b: 26 };
    const xs = (s) => pad.l + ((Math.log2(s) - 10) / 20) * (W - pad.l - pad.r);
    const series = sizes.map((s) => ({ s, ...scanTimes(s, p) }));
    let lo = Infinity;
    let hi = 0;
    series.forEach((d) => {
      [d.ring, d.tree].forEach((t) => {
        lo = Math.min(lo, t);
        hi = Math.max(hi, t);
      });
    });
    const logLo = Math.log10(lo);
    const logHi = Math.log10(hi);
    const ys = (t) => pad.t + (1 - (Math.log10(t) - logLo) / (logHi - logLo)) * (H - pad.t - pad.b);
    const line = (key) =>
      '<polyline class="nccl-line-' + key + '" points="' +
      series.map((d) => xs(d.s).toFixed(1) + "," + ys(d[key]).toFixed(1)).join(" ") + '"></polyline>';
    let grid = "";
    [KIB, MIB, GIB].forEach((s) => {
      grid += '<line class="nccl-grid" x1="' + xs(s) + '" y1="' + pad.t + '" x2="' + xs(s) + '" y2="' + (H - pad.b) +
        '"></line><text class="nccl-tick" x="' + xs(s) + '" y="' + (H - 8) + '" text-anchor="middle">' +
        (s >= GIB ? "1 GiB" : s >= MIB ? "1 MiB" : "1 KiB") + "</text>";
    });
    const ticks = [lo, Math.sqrt(lo * hi), hi];
    ticks.forEach((t) => {
      grid += '<line class="nccl-grid" x1="' + pad.l + '" y1="' + ys(t) + '" x2="' + (W - pad.r) + '" y2="' + ys(t) +
        '"></line><text class="nccl-tick" x="' + (pad.l - 5) + '" y="' + (ys(t) + 3) + '" text-anchor="end">' +
        fmtTime(t) + "</text>";
    });
    let marker = "";
    if (cross > 0 && cross >= KIB && cross <= GIB) {
      marker = '<line class="nccl-cross" x1="' + xs(cross) + '" y1="' + pad.t + '" x2="' + xs(cross) +
        '" y2="' + (H - pad.b) + '"></line>';
    }
    return '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="ring vs tree time">' +
      grid + marker + line("tree") + line("ring") + "</svg>";
  }

  function busbwChartSvg(sizes, p, cross) {
    const W = 340;
    const H = 190;
    const pad = { l: 62, r: 18, t: 8, b: 26 };
    const xs = (s) => pad.l + ((Math.log2(s) - 10) / 20) * (W - pad.l - pad.r);
    const cap = p.bw;
    const ys = (b) => pad.t + (1 - b / cap) * (H - pad.t - pad.b);
    const series = sizes.map((s) => {
      const t = scanTimes(s, p);
      return { s, ring: (2 * (p.n - 1) / p.n) * (s / t.ring), tree: (2 * (p.n - 1) / p.n) * (s / t.tree) };
    });
    const line = (key) =>
      '<polyline class="nccl-line-' + key + '" points="' +
      series.map((d) => xs(d.s).toFixed(1) + "," + Math.max(0, ys(d[key])).toFixed(1)).join(" ") + '"></polyline>';
    let grid = "";
    [KIB, MIB, GIB].forEach((s) => {
      grid += '<line class="nccl-grid" x1="' + xs(s) + '" y1="' + pad.t + '" x2="' + xs(s) + '" y2="' + (H - pad.b) +
        '"></line><text class="nccl-tick" x="' + xs(s) + '" y="' + (H - 8) + '" text-anchor="middle">' +
        (s >= GIB ? "1 GiB" : s >= MIB ? "1 MiB" : "1 KiB") + "</text>";
    });
    [0, cap / 2, cap].forEach((b) => {
      grid += '<line class="nccl-grid" x1="' + pad.l + '" y1="' + ys(b) + '" x2="' + (W - pad.r) + '" y2="' + ys(b) +
        '"></line><text class="nccl-tick" x="' + (pad.l - 5) + '" y="' + (ys(b) + 3) + '" text-anchor="end">' +
        (b / 1e9).toFixed(0) + "</text>";
    });
    let marker = "";
    if (cross > 0 && cross >= KIB && cross <= GIB) {
      marker = '<line class="nccl-cross" x1="' + xs(cross) + '" y1="' + pad.t + '" x2="' + xs(cross) +
        '" y2="' + (H - pad.b) + '"></line>';
    }
    return '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="ring vs tree busbw">' +
      grid + marker + line("tree") + line("ring") + "</svg>";
  }

  /* ============================================ topology / protocol */

  function mountTopo(host) {
    if (!host) return;
    host.innerHTML =
      '<div class="sim-controls nccl-topo-controls">' +
      control("链路档位", '<select data-nccl-tp="link">' +
        LINKS.map((l) => option(l.id, l.name, topo.link)).join("") + "</select>") +
      control("Rank 数", '<select data-nccl-tp="n">' +
        [2, 4, 8].map((v) => option(v, v + " 卡", topo.n)).join("") + "</select>") +
      control("消息大小 S", '<select data-nccl-tp="size">' +
        TOPO_SIZES.map((s, i) => option(i, s.label, topo.sizeIdx)).join("") + "</select>") +
      "</div>" +
      '<div class="compare" data-nccl-tp="stats"></div>' +
      '<div class="table-wrap"><table class="tagtable nccl-topo-table">' +
      "<thead><tr><th>链路</th><th>BW</th><th>α</th><th>ring 估算</th><th>tree 估算</th><th>建议</th></tr></thead>" +
      '<tbody data-nccl-tp="table"></tbody></table></div>' +
      '<p class="note" data-nccl-tp="advice"></p>';

    host.querySelectorAll("select[data-nccl-tp]").forEach((inp) => {
      inp.addEventListener("change", () => {
        const f = inp.dataset.ncclTp;
        if (f === "link") topo.link = inp.value;
        else if (f === "n") topo.n = +inp.value;
        else if (f === "size") topo.sizeIdx = +inp.value;
        updateTopo(host);
      });
    });
    updateTopo(host);
  }

  function topoLink() {
    return LINKS.find((l) => l.id === topo.link) || LINKS[5];
  }

  function topoParams() {
    const link = topoLink();
    const n = topo.n;
    const size = TOPO_SIZES[topo.sizeIdx] ? TOPO_SIZES[topo.sizeIdx].bytes : 128 * MIB;
    const bw = link.gbs * 1e9;
    const lat = link.latUs * 1e-6;
    const levels = treeLevels(n);
    return {
      link,
      n,
      size,
      levels,
      ring: 2 * (n - 1) * (lat + size / n / bw),
      tree: 2 * levels * (lat + size / bw),
      perRank: (2 * (n - 1) / n) * size,
      busbw: link.gbs,
    };
  }

  function protocolFor(link, size) {
    if (size <= 16 * KIB) return "LL";
    if (size <= 2 * MIB) return "LL128";
    if (link.id === "nvlink" || link.id === "nvswitch") return "Simple（可试 NVLS）";
    return "Simple";
  }

  function updateTopo(host) {
    if (!host) return;
    const p = topoParams();
    const winner = p.tree < p.ring ? "tree" : "ring";
    const stats = host.querySelector('[data-nccl-tp="stats"]');
    if (stats) {
      stats.innerHTML =
        '<div class="stat"><div class="stat-label">busbw 上限</div><div class="stat-value">' +
        p.busbw.toFixed(0) + ' GB/s</div><div class="stat-sub">ring 大消息时逼近单链路带宽</div></div>' +
        '<div class="stat"><div class="stat-label">ring 步数</div><div class="stat-value">' +
        (2 * (p.n - 1)) + ' 步</div><div class="stat-sub">tree ' + (2 * p.levels) + " 步</div></div>" +
        '<div class="stat"><div class="stat-label">每卡总发出</div><div class="stat-value">' +
        fmtBytes(p.perRank) + '</div><div class="stat-sub">2(N−1)/N × S</div></div>' +
        '<div class="stat"><div class="stat-label">ring 估算</div><div class="stat-value">' +
        fmtTime(p.ring) + '</div><div class="stat-sub">tree ' + fmtTime(p.tree) + "；当前 " + winner + " 更快</div></div>";
    }
    const table = host.querySelector('[data-nccl-tp="table"]');
    if (table) {
      table.innerHTML = LINKS.map((l) => {
        const bw = l.gbs * 1e9;
        const lat = l.latUs * 1e-6;
        const rt = 2 * (p.n - 1) * (lat + p.size / p.n / bw);
        const tt = 2 * p.levels * (lat + p.size / bw);
        const w = tt < rt ? "tree" : "ring";
        return (
          "<tr" + (l.id === p.link.id ? ' class="on"' : "") + "><td><b>" + esc(l.name) + "</b></td><td>" +
          l.gbs + " GB/s</td><td>" + l.latUs + " µs</td><td>" + fmtTime(rt) + "</td><td>" +
          fmtTime(tt) + "</td><td>" + w + " + " + esc(protocolFor(l, p.size)) + "</td></tr>"
        );
      }).join("");
    }
    const advice = host.querySelector('[data-nccl-tp="advice"]');
    if (advice) {
      advice.innerHTML =
        "<b>" + esc(p.link.name) + "</b>（" + p.link.note + "）：S = " + fmtBytes(p.size) + "，N = " + p.n + "；" +
        "当前建议 <b>" + winner + " + " + esc(protocolFor(p.link, p.size)) + "</b>。" +
        (p.size <= 16 * KIB
          ? " 这是 decode TP 的小消息区，α 占主导：优先同机 NVLink 与低延迟协议，不要指望提高 BW 能救。"
          : p.size <= 2 * MIB
            ? " 中等消息落在 LL128 / tree 与 ring 的交界附近，按实测 busbw 选。"
            : " 大消息进入带宽区：ring + Simple，确保 busbw 收敛到链路带宽。");
    }
  }

  /* ----------------------------------------------------------- helpers */

  function section(sec, body) {
    return '<h2 class="sec-title">' + esc(sec.title) + "</h2>" +
      '<p class="page-lead">' + esc(sec.lead) + "</p>" + body;
  }

  function takeaway(sec) {
    const t = sec && sec.takeaway;
    if (!t) return "";
    return '<div class="nccl-take">' +
      [["为什么", t.why], ["怎么算", t.calc], ["推理落点", t.infer]]
        .filter((x) => x[1])
        .map((x) => '<div class="nccl-take-card"><b>' + x[0] + "</b><p>" + esc(x[1]) + "</p></div>")
        .join("") +
      "</div>";
  }

  function control(label, input) {
    return '<label class="sim-field"><span>' + esc(label) + "</span>" + input + "</label>";
  }

  function option(value, label, current) {
    return '<option value="' + esc(value) + '"' + (String(value) === String(current) ? " selected" : "") + ">" +
      esc(label) + "</option>";
  }

  function row(label, value, sub) {
    return '<div class="sim-metric"><span>' + esc(label) + "</span><b>" + esc(value) +
      (sub ? "<em>" + esc(sub) + "</em>" : "") + "</b></div>";
  }

  function details(q, a) {
    return '<details class="thinkbox"><summary>' + esc(q) + "</summary><p>" + esc(a) + "</p></details>";
  }

  function fmtBytes(b) {
    if (!isFinite(b)) return "—";
    if (b >= GIB) return (b / GIB).toFixed(2) + " GiB";
    if (b >= MIB) return (b / MIB).toFixed(1) + " MiB";
    if (b >= KIB) return (b / KIB).toFixed(1) + " KiB";
    return Math.round(b) + " B";
  }

  function fmtTime(s) {
    if (!isFinite(s)) return "—";
    if (s >= 1) return s.toFixed(3) + " s";
    if (s >= 1e-3) return (s * 1e3).toFixed(3) + " ms";
    return (s * 1e6).toFixed(2) + " µs";
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
})();
