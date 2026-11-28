/* Document chapter: ECN + DCQCN + PFC for lossless RoCEv2.
 * Content lives in web/tools/ecn.json; example/ecn_sim.py is injected by
 * build.py as data.simCode + data.simOutput.
 *
 * Interactives (state local to this IIFE, listeners bound to the current host,
 * ids prefixed ecn-):
 *   1. ECN/WRED threshold sliders -> WRED curve + queue/rate convergence
 *   2. PFC / ECN / combined scenario picker -> drops, pauses, throughput
 *   3. DSCP -> TC -> PFC/ECN mapping picker, per traffic type
 *   4. leaf-spine vs rail-optimized topology diagram switch
 */
(function () {
  const LK = (window.LK = window.LK || {});
  const E = (LK.ecn = LK.ecn || {});

  // Same simplified model as example/ecn_sim.py (normalized MB/tick).
  const DEFAULT_CFG = {
    link: 10, buf: 20, xoff: 6, xon: 3, r0: 12.5,
    rmin: 8.5, rmax: 12.5, rtt: 6, g: 0.0625, rai: 0.15,
    cnpInterval: 2, maxp: 0.2, ticks: 2000,
  };
  const PKT_MB = 0.5;
  const SIM_TICKS = 300; // interactive preview window

  const wredState = { min: 2, max: 5, maxp: 0.2 };
  const scenState = { idx: 2 };
  const mapState = { idx: 0 };
  const topoState = { idx: 0 };

  let timer = null;

  E.stop = function () {
    if (timer) clearInterval(timer);
    timer = null;
  };

  E.render = function (el, data, world) {
    E.stop();
    if (!el) return;
    data = data || {};
    el.innerHTML =
      '<p class="kicker">' + esc(data.kicker) + "</p>" +
      '<h1 class="page-title">' + esc(data.title) + "</h1>" +
      '<p class="page-lead">' + esc(data.lead) + "</p>" +
      whyHTML(data.why) +
      layersHTML(data.layers) +
      switchHTML(data.switch) +
      serverHTML(data.server) +
      losslessHTML(data.lossless) +
      priorityHTML(data.priority) +
      ecnHTML(data.ecn) +
      tuningHTML(data.tuning) +
      codeHTML(data.codeSection, data.simOutput) +
      checkHTML(data.check);

    mountWred(el.querySelector("#ecn-wred"), data.ecn);
    mountScenario(el.querySelector("#ecn-scenario"), data.lossless);
    mountMapping(el.querySelector("#ecn-mapping"), data.priority);
    mountTopo(el.querySelector("#ecn-topo"), data.switch);

    const codeEl = el.querySelector("#ecn-sim-code");
    if (codeEl && LK.codeview && data.simCode) LK.codeview.mount(codeEl, data.simCode.lines);
  };

  /* ------------------------------------------------------------- sections */

  function whyHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="net-cards">' +
      arr(sec.points).map((x, i) =>
        '<div class="net-card"><span class="train-no">0' + (i + 1) + "</span><b>" + esc(x.name) +
        "</b><p>" + esc(x.detail) + "</p></div>"
      ).join("") + "</div>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function layersHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="ecn-layers">' +
      arr(sec.steps).map((s) =>
        '<div class="ecn-layer"><b>' + esc(s.step) + '</b><span class="ecn-where">' + esc(s.where) +
        "</span><p>" + esc(s.what) + "</p></div>"
      ).join("") + "</div>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function switchHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<h3 class="sec-sub">' + esc(sec.topoTitle) + "</h3>" +
      '<div id="ecn-topo" class="ecn-topo-host"></div>' +
      '<p class="note">' + esc(sec.topoNote) + "</p>" +
      '<h3 class="sec-sub">3.2 配置步骤（vendor-neutral）</h3>' +
      '<ol class="ecn-steps">' + arr(sec.steps).map((s) => "<li>" + esc(s) + "</li>").join("") + "</ol>" +
      '<h3 class="sec-sub">3.3 交换机命令示意</h3>' +
      arr(sec.vendors).map((v) =>
        '<div class="ecn-vendor"><b>' + esc(v.vendor) + "</b>" +
        '<pre class="train-run"><code>' + esc(arr(v.commands).join("\n")) + "</code></pre></div>"
      ).join("") +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function serverHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      arr(sec.steps).map((s) =>
        '<div class="ecn-server-step"><b>' + esc(s.step) + "</b><p>" + esc(s.detail) + "</p>" +
        '<pre class="train-run"><code>' + esc(arr(s.commands).join("\n")) + "</code></pre></div>"
      ).join("") +
      '<h3 class="sec-sub">' + esc(sec.envTitle) + "</h3>" +
      '<div class="table-wrap"><table class="tagtable ecn-env-table"><thead><tr>' +
      "<th>环境变量</th><th>示例值</th><th>作用 / 对齐点</th>" +
      "</tr></thead><tbody>" +
      arr(sec.envs).map((e) =>
        "<tr><td><code>" + esc(e.name) + "</code></td><td><code>" + esc(e.value) +
        "</code></td><td>" + esc(e.why) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function losslessHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<ul class="train-rules ecn-principles">' +
      arr(sec.principles).map((p) => "<li>" + esc(p) + "</li>").join("") + "</ul>" +
      '<div class="ecn-callout"><b>阈值关系</b><p>' + esc(sec.thresholdText) + "</p></div>" +
      '<h3 class="sec-sub">PFC 的风险</h3>' +
      '<ul class="train-rules ecn-risks">' + arr(sec.risks).map((r) => "<li>" + esc(r) + "</li>").join("") + "</ul>" +
      '<h3 class="sec-sub">' + esc(sec.scenarioTitle) + "</h3>" +
      '<div id="ecn-scenario" class="ecn-scenario-host"></div>' +
      '<p class="note">' + esc(sec.scenarioNote) + "</p>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function priorityHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="table-wrap"><table class="tagtable ecn-map-table"><thead><tr>' +
      "<th>业务</th><th>DSCP</th><th>TC</th><th>PFC</th><th>ECN</th><th>说明</th>" +
      "</tr></thead><tbody>" +
      arr(sec.mapping).map((r) =>
        "<tr><td><b>" + esc(r.traffic) + "</b></td><td>" + esc(r.dscp) + "</td><td>" + esc(r.tc) +
        "</td><td>" + esc(r.pfc) + "</td><td>" + esc(r.ecn) + "</td><td>" + esc(r.note) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<h3 class="sec-sub">' + esc(sec.mappingTitle) + "</h3>" +
      '<div id="ecn-mapping" class="ecn-mapping-host"></div>' +
      '<p class="note">' + esc(sec.mappingNote) + "</p>" +
      '<h3 class="sec-sub">优先级设计规则</h3>' +
      '<ul class="train-rules">' + arr(sec.rules).map((r) => "<li>" + esc(r) + "</li>").join("") + "</ul>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function ecnHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<ul class="train-rules ecn-principles">' +
      arr(sec.design).map((d) => "<li>" + esc(d) + "</li>").join("") + "</ul>" +
      '<h3 class="sec-sub">7.1 示例参数</h3>' +
      '<div class="table-wrap"><table class="tagtable ecn-param-table"><thead><tr>' +
      "<th>参数</th><th>示例值</th><th>为什么</th>" +
      "</tr></thead><tbody>" +
      arr(sec.params).map((p) =>
        "<tr><td><b>" + esc(p.name) + "</b></td><td>" + esc(p.value) + "</td><td>" + esc(p.why) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<h3 class="sec-sub">' + esc(sec.calcTitle) + "</h3>" +
      '<div id="ecn-wred" class="ecn-wred-host"></div>' +
      '<p class="note">' + esc(sec.calcNote) + "</p>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function tuningHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="table-wrap"><table class="tagtable ecn-tuning-table"><thead><tr>' +
      "<th>命令 / 指标</th><th>看什么</th>" +
      "</tr></thead><tbody>" +
      arr(sec.commands).map((c) =>
        "<tr><td><code>" + esc(c.cmd) + "</code></td><td>" + esc(c.what) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<h3 class="sec-sub">验收清单</h3>' +
      '<ul class="train-rules">' + arr(sec.checks).map((c) => "<li>" + esc(c) + "</li>").join("") + "</ul>" +
      '<h3 class="sec-sub">常见坑</h3>' +
      '<ul class="train-rules ecn-pitfalls">' + arr(sec.pitfalls).map((p) => "<li>" + esc(p) + "</li>").join("") + "</ul>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function codeHTML(sec, simOutput) {
    sec = sec || {};
    return sectionHead(sec) +
      '<pre class="train-run"><code>' + esc(sec.run) + "</code></pre>" +
      '<ul class="train-rules">' + arr(sec.notes).map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul>" +
      '<div class="train-code-head"><span>' + esc(sec.file) + "</span><span>纯 Python · 可运行</span></div>" +
      '<div class="train-code" id="ecn-sim-code"></div>' +
      '<h3 class="sec-sub">示例输出</h3>' +
      '<pre class="train-run"><code>' + esc(simOutput) + "</code></pre>";
  }

  function checkHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<ul class="train-rules train-checklist">' +
      arr(sec.checklist).map((c) => "<li>" + esc(c) + "</li>").join("") + "</ul>" +
      arr(sec.questions).map((q) => details(q.q, q.a)).join("");
  }

  /* -------------------------------------------------------- 1. WRED + sim */

  function mountWred(host, ecnSec) {
    if (!host) return;
    const cfg = simCfg(ecnSec);
    host.innerHTML =
      '<div class="ecn-controls">' +
      ctl("ECN min", '<input type="range" data-wred="min" min="0.5" max="8" step="0.5" value="' +
        wredState.min + '"><b data-wred-val="min">' + wredState.min + " MB</b>") +
      ctl("ECN max", '<input type="range" data-wred="max" min="1" max="14" step="0.5" value="' +
        wredState.max + '"><b data-wred-val="max">' + wredState.max + " MB</b>") +
      ctl("max_p", '<input type="range" data-wred="maxp" min="0.01" max="0.6" step="0.01" value="' +
        wredState.maxp + '"><b data-wred-val="maxp">' + (wredState.maxp * 100).toFixed(0) + "%</b>") +
      "</div>" +
      '<div id="ecn-wred-body"></div>';
    host.querySelectorAll("[data-wred]").forEach((inp) => {
      inp.addEventListener("input", () => {
        const f = inp.dataset.wred;
        const v = parseFloat(inp.value);
        if (f === "min") wredState.min = v;
        else if (f === "max") wredState.max = v;
        else wredState.maxp = v;
        clampWred(host);
        updateWred(host, cfg);
      });
    });
    updateWred(host, cfg);
  }

  function clampWred(host) {
    if (wredState.max < wredState.min + 0.5) wredState.max = wredState.min + 0.5;
    if (wredState.max > 14) wredState.max = 14;
    host.querySelectorAll("[data-wred]").forEach((inp) => {
      const f = inp.dataset.wred;
      const v = f === "min" ? wredState.min : f === "max" ? wredState.max : wredState.maxp;
      inp.value = String(v);
      const label = host.querySelector('[data-wred-val="' + f + '"]');
      if (label) {
        label.textContent = f === "maxp" ? (v * 100).toFixed(0) + "%" : v + " MB";
      }
    });
  }

  function simCfg(ecnSec) {
    const s = (ecnSec && ecnSec.sim) || {};
    const cfg = {};
    Object.keys(DEFAULT_CFG).forEach((k) => {
      cfg[k] = typeof s[k] === "number" ? s[k] : DEFAULT_CFG[k];
    });
    return cfg;
  }

  function updateWred(host, cfg) {
    const body = host.querySelector("#ecn-wred-body");
    if (!body) return;
    const sim = ecnSim(cfg, wredState.min, wredState.max, wredState.maxp, SIM_TICKS);
    const early = wredState.max < cfg.xoff && wredState.min < cfg.xoff;
    const tone = early ? "ok" : "bad";
    body.innerHTML =
      '<div class="ecn-wred">' + wredSvg(wredState.min, wredState.max, wredState.maxp, cfg) + "</div>" +
      '<div class="compare">' +
      stat("ECN 区间", wredState.min + "–" + wredState.max + " MB", "min→max 按概率标记") +
      stat("相对 PFC XOFF", '<span class="ecn-tone ' + tone + '">' +
        (early ? "ECN 更早（推荐）" : "ECN 不早于 PFC") + "</span>",
        "ECN max 应 < XOFF " + cfg.xoff + " MB") +
      stat("模拟标记率", (sim.markRate * 100).toFixed(1) + "%", "前 " + SIM_TICKS + " tick") +
      stat("模拟平均队列", sim.avgQueue.toFixed(2) + " MB", "PFC pause " + sim.pauses + " 次") +
      "</div>" +
      '<div class="ecn-trace">' + traceSvg(sim, cfg) + "</div>" +
      '<p class="note">' +
      (early
        ? "ECN 先于 PFC：队列到 " + wredState.min + " MB 开始按概率标记，到 " + wredState.max +
          " MB 全标；发送端收到 CNP 后降速，PFC 只在尖峰兜底。"
        : "当前 ECN max ≥ PFC XOFF（" + cfg.xoff + " MB）：队列会先触发 PFC 暂停，ECN 基本来不及生效；把 max 降到 XOFF 以下。") +
      "</p>";
  }

  function wredSvg(emin, emax, maxp, cfg) {
    const W = 560, H = 240, left = 46, right = 540, top = 14, bottom = 208;
    const xOf = (q) => left + Math.max(0, Math.min(1, q / cfg.buf)) * (right - left);
    const yOf = (p) => bottom - Math.max(0, Math.min(1, p)) * (bottom - top);
    const xticks = [0, 5, 10, 15, 20];
    const yticks = [0, 0.25, 0.5, 0.75, 1];
    let grid = "";
    xticks.forEach((q) => {
      grid += '<line class="ecn-wred-grid" x1="' + xOf(q) + '" y1="' + top + '" x2="' + xOf(q) +
        '" y2="' + bottom + '"></line><text class="ecn-wred-label" x="' + xOf(q) + '" y="' + (bottom + 15) +
        '" text-anchor="middle">' + q + "</text>";
    });
    yticks.forEach((p) => {
      grid += '<line class="ecn-wred-grid" x1="' + left + '" y1="' + yOf(p) + '" x2="' + right +
        '" y2="' + yOf(p) + '"></line><text class="ecn-wred-label" x="' + (left - 6) + '" y="' +
        (yOf(p) + 4) + '" text-anchor="end">' + p + "</text>";
    });
    const points = [[0, 0], [emin, 0], [emax, maxp], [emax, 1], [cfg.buf, 1]]
      .map((p) => xOf(p[0]).toFixed(1) + "," + yOf(p[1]).toFixed(1)).join(" ");
    const minLine = '<line class="ecn-wred-mark" x1="' + xOf(emin) + '" y1="' + top + '" x2="' + xOf(emin) +
      '" y2="' + bottom + '"></line><text class="ecn-wred-label" x="' + xOf(emin) + '" y="' + (top + 10) +
      '" text-anchor="middle">min</text>';
    const maxLine = '<line class="ecn-wred-mark" x1="' + xOf(emax) + '" y1="' + top + '" x2="' + xOf(emax) +
      '" y2="' + bottom + '"></line><text class="ecn-wred-label" x="' + xOf(emax) + '" y="' + (top + 10) +
      '" text-anchor="middle">max</text>';
    const xoffLine = '<line class="ecn-wred-xoff" x1="' + xOf(cfg.xoff) + '" y1="' + top + '" x2="' +
      xOf(cfg.xoff) + '" y2="' + bottom + '"></line><text class="ecn-wred-label xoff" x="' +
      (xOf(cfg.xoff) - 4) + '" y="' + (bottom - 6) + '" text-anchor="end">XOFF ' + cfg.xoff + "</text>";
    return '<svg class="ecn-wred-svg" viewBox="0 0 ' + W + " " + H +
      '" role="img" aria-label="WRED marking probability over queue length">' +
      grid + minLine + maxLine + xoffLine +
      '<polyline class="ecn-wred-curve" points="' + points + '"></polyline>' +
      '<text class="ecn-wred-axis" x="' + right + '" y="' + (H - 5) +
      '" text-anchor="end">queue（MB）</text>' +
      '<text class="ecn-wred-axis" x="8" y="13">mark p</text></svg>';
  }

  function traceSvg(sim, cfg) {
    const W = 560, H = 170, left = 46, right = 540, top = 14, bottom = 138;
    const n = sim.queue.length;
    const xOf = (i) => left + (i / Math.max(1, n - 1)) * (right - left);
    const yQ = (v) => bottom - Math.max(0, Math.min(1, v / cfg.buf)) * (bottom - top);
    const yR = (v) => bottom - Math.max(0, Math.min(1, v / cfg.rmax)) * (bottom - top);
    const queuePts = sim.queue.map((v, i) => xOf(i).toFixed(1) + "," + yQ(v).toFixed(1)).join(" ");
    const ratePts = sim.rate.map((v, i) => xOf(i).toFixed(1) + "," + yR(v).toFixed(1)).join(" ");
    const yM = (v) => bottom - Math.max(0, Math.min(1, v)) * (bottom - top);
    const markPts = (sim.markTrace || []).map((v, i) => xOf(i).toFixed(1) + "," + yM(v).toFixed(1)).join(" ");
    const xoffY = yQ(cfg.xoff);
    return '<svg class="ecn-trace-svg" viewBox="0 0 ' + W + " " + H +
      '" role="img" aria-label="queue and rate over 300 ticks">' +
      '<line class="ecn-trace-grid" x1="' + left + '" y1="' + xoffY + '" x2="' + right + '" y2="' +
      xoffY + '"></line>' +
      '<text class="ecn-trace-label" x="' + (left + 4) + '" y="' + (xoffY - 4) + '">XOFF ' + cfg.xoff + "</text>" +
      '<line class="ecn-trace-axis" x1="' + left + '" y1="' + bottom + '" x2="' + right + '" y2="' + bottom + '"></line>' +
      '<line class="ecn-trace-axis" x1="' + left + '" y1="' + top + '" x2="' + left + '" y2="' + bottom + '"></line>' +
      '<polyline class="ecn-trace-queue" points="' + queuePts + '"></polyline>' +
      '<polyline class="ecn-trace-rate" points="' + ratePts + '"></polyline>' +
      '<polyline class="ecn-trace-mark" points="' + markPts + '"></polyline>' +
      '<text class="ecn-trace-label queue" x="' + (left + 6) + '" y="' + (top + 10) + '">queue</text>' +
      '<text class="ecn-trace-label rate" x="' + (left + 66) + '" y="' + (top + 10) + '">rate</text>' +
      '<text class="ecn-trace-label mark" x="' + (left + 112) + '" y="' + (top + 10) + '">mark</text>' +
      '<text class="ecn-trace-axis" x="' + right + '" y="' + (H - 5) +
      '" text-anchor="end">tick（前 ' + SIM_TICKS + "）</text></svg>";
  }

  /* simplified ECN/PFC sim for the interactive preview (same shape as Python) */
  function ecnSim(cfg, eminRaw, emaxRaw, maxp, ticks) {
    const emin = Math.max(0, eminRaw);
    const emax = Math.max(emin + 0.01, emaxRaw);
    let seed = 12345 >>> 0;
    const rnd = () => {
      seed ^= seed << 13;
      seed >>>= 0;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      seed >>>= 0;
      return seed / 4294967296;
    };
    let q = 0, rate = cfg.r0, alpha = 0, paused = false, pauseLeft = 0;
    let pauses = 0, drops = 0, delivered = 0, sent = 0, marked = 0, qSum = 0, pktAcc = 0;
    let lastCnp = -1e9;
    const cnp = new Array(ticks + cfg.rtt + 2).fill(0);
    const queue = [], rateTrace = [], markTrace = [];
    for (let t = 0; t < ticks; t++) {
      if (cnp[t]) {
        alpha = (1 - cfg.g) * alpha + cfg.g;
        rate = Math.max(cfg.rmin, rate * (1 - alpha / 2));
      } else {
        alpha = (1 - cfg.g) * alpha;
        if (!paused) rate = Math.min(cfg.rmax, rate + cfg.rai);
      }
      const send = paused ? 0 : rate;
      const qEcn = q;
      const space = cfg.buf - q;
      let enq = send;
      if (send >= space) {
        drops += send - space;
        enq = space;
      }
      q += enq;
      pktAcc += enq;
      let marks = 0, tickPkts = 0;
      while (pktAcc + 1e-12 >= PKT_MB) {
        pktAcc -= PKT_MB;
        sent += 1;
        tickPkts += 1;
        let p = 0;
        if (qEcn >= emax) p = 1;
        else if (qEcn >= emin) p = ((qEcn - emin) / (emax - emin)) * maxp;
        if (rnd() < p) {
          marks += 1;
          marked += 1;
        }
      }
      if (marks && t - lastCnp >= cfg.cnpInterval) {
        cnp[Math.min(t + cfg.rtt, cnp.length - 1)] = 1;
        lastCnp = t;
      }
      const served = Math.min(q, cfg.link);
      q -= served;
      delivered += served;
      qSum += q;
      queue.push(q);
      rateTrace.push(send);
      markTrace.push(tickPkts ? marks / tickPkts : 0);
      if (paused) {
        pauseLeft -= 1;
        if (pauseLeft <= 0) {
          if (q <= cfg.xon) paused = false;
          else pauseLeft = 1;
        }
      } else if (q >= cfg.xoff) {
        paused = true;
        pauseLeft = 1;
        pauses += 1;
      }
    }
    return {
      queue,
      rate: rateTrace,
      markRate: marked / Math.max(1, sent),
      markTrace,
      avgQueue: qSum / ticks,
      throughput: delivered / ticks,
      pauses,
      drops,
    };
  }

  /* ------------------------------------------------- 2. scenario picker */

  function mountScenario(host, lossless) {
    if (!host) return;
    lossless = lossless || {};
    const scenarios = arr(lossless.scenarios);
    if (!scenarios.length) return;
    host.innerHTML =
      '<div class="ecn-scenarios">' +
      scenarios.map((s, i) =>
        '<button type="button" class="ecn-scenario' + (i === scenState.idx ? " on" : "") +
        '" data-scen="' + i + '"><b>' + esc(s.name) + '</b><span class="ecn-badge ' +
        (s.pfc === "on" && s.ecn === "on" ? "both" : s.pfc === "on" ? "pfc" : "ecn") + '">' +
        "PFC " + esc(s.pfc) + " / ECN " + esc(s.ecn) + "</span></button>"
      ).join("") + "</div>" +
      '<div id="ecn-scen-body"></div>';
    host.querySelectorAll("[data-scen]").forEach((b) => {
      b.addEventListener("click", () => {
        scenState.idx = +b.dataset.scen;
        host.querySelectorAll("[data-scen]").forEach((x) =>
          x.classList.toggle("on", +x.dataset.scen === scenState.idx)
        );
        updateScenario(host, lossless);
      });
    });
    updateScenario(host, lossless);
  }

  function updateScenario(host, lossless) {
    const body = host.querySelector("#ecn-scen-body");
    if (!body) return;
    const scenarios = arr((lossless || {}).scenarios);
    const s = scenarios[scenState.idx] || scenarios[0];
    if (!s) {
      body.innerHTML = "";
      return;
    }
    body.innerHTML =
      '<div class="compare">' +
      stat("丢包", s.drops, s.drops === "0.0 MB" ? "PFC 兜底成功" : "buffer 溢出（无 PFC）") +
      stat("吞吐", s.throughput, "归一化账本；线速约 10") +
      stat("平均队列", s.avgq, "队列越低，延迟越稳") +
      stat("ECN 标记率", s.mark, "过高说明降速过猛") +
      stat("PFC pause", s.pauses, "次数越多，抖动/PFC storm 风险越高") +
      "</div>" +
      '<div class="ecn-scen-result"><span class="ecn-badge ' +
      (s.pfc === "on" && s.ecn === "on" ? "both" : s.pfc === "on" ? "pfc" : "ecn") + '">' +
      "PFC " + esc(s.pfc) + " · ECN " + esc(s.ecn) + "</span><p>" + esc(s.reading) + "</p></div>";
  }

  /* ------------------------------------------------- 3. DSCP mapping */

  function mountMapping(host, priority) {
    if (!host) return;
    priority = priority || {};
    const rows = arr(priority.mapping);
    if (!rows.length) return;
    host.innerHTML =
      '<div class="ecn-mapping-buttons">' +
      rows.map((r, i) =>
        '<button type="button" class="ecn-map-btn' + (i === mapState.idx ? " on" : "") +
        '" data-map="' + i + '">' + esc(r.traffic) + "</button>"
      ).join("") + "</div>" +
      '<div id="ecn-map-body"></div>';
    host.querySelectorAll("[data-map]").forEach((b) => {
      b.addEventListener("click", () => {
        mapState.idx = +b.dataset.map;
        host.querySelectorAll("[data-map]").forEach((x) =>
          x.classList.toggle("on", +x.dataset.map === mapState.idx)
        );
        updateMapping(host, priority);
      });
    });
    updateMapping(host, priority);
  }

  function updateMapping(host, priority) {
    const body = host.querySelector("#ecn-map-body");
    if (!body) return;
    const rows = arr((priority || {}).mapping);
    const r = rows[mapState.idx] || rows[0];
    if (!r) {
      body.innerHTML = "";
      return;
    }
    body.innerHTML =
      '<div class="ecn-map-result">' +
      '<div class="ecn-map-fields">' +
      field("DSCP", r.dscp) + field("TC / 队列", r.tc) +
      field("PFC", r.pfc) + field("ECN", r.ecn) +
      "</div>" +
      '<p class="note">' + esc(r.note) + "</p>" +
      "</div>";
  }

  function field(label, value) {
    return '<div class="ecn-map-field"><span>' + esc(label) + "</span><b>" + esc(value) + "</b></div>";
  }

  /* -------------------------------------------------- 4. topology switch */

  function mountTopo(host, switchSec) {
    if (!host) return;
    switchSec = switchSec || {};
    const topologies = arr(switchSec.topologies);
    if (!topologies.length) return;
    host.innerHTML =
      '<div class="ecn-topo-buttons">' +
      topologies.map((t, i) =>
        '<button type="button" class="ecn-topo-btn' + (i === topoState.idx ? " on" : "") +
        '" data-topo="' + i + '">' + esc(t.name) + "</button>"
      ).join("") + "</div>" +
      '<div id="ecn-topo-body"></div>';
    host.querySelectorAll("[data-topo]").forEach((b) => {
      b.addEventListener("click", () => {
        topoState.idx = +b.dataset.topo;
        host.querySelectorAll("[data-topo]").forEach((x) =>
          x.classList.toggle("on", +x.dataset.topo === topoState.idx)
        );
        updateTopo(host, switchSec);
      });
    });
    updateTopo(host, switchSec);
  }

  function updateTopo(host, switchSec) {
    const body = host.querySelector("#ecn-topo-body");
    if (!body) return;
    const topologies = arr((switchSec || {}).topologies);
    const t = topologies[topoState.idx] || topologies[0];
    if (!t) {
      body.innerHTML = "";
      return;
    }
    body.innerHTML =
      '<div class="ecn-topo-diagram">' +
      (topoState.idx === 0 ? leafSpineSvg() : railSvg()) +
      "</div>" +
      '<div class="ecn-topo-result"><b>' + esc(t.name) + "</b><p>" + esc(t.detail) + "</p></div>";
  }

  function leafSpineSvg() {
    const spines = [90, 250, 410].map((x, i) => node(x, 24, 110, 26, "spine " + (i + 1), "spine"));
    const leaves = [30, 165, 300, 435].map((x, i) => node(x, 118, 100, 26, "leaf " + (i + 1), "leaf"));
    let links = "";
    [90, 250, 410].forEach((sx) => {
      [80, 215, 350, 485].forEach((lx) => {
        links += '<line class="ecn-topo-link" x1="' + sx + '" y1="52" x2="' + lx + '" y2="118"></line>';
      });
    });
    const hosts = [80, 215, 350, 485].map((x, i) => node(x - 34, 188, 68, 24, "host " + (i + 1), "host"));
    let hostLinks = "";
    [80, 215, 350, 485].forEach((x) => {
      hostLinks += '<line class="ecn-topo-link dim" x1="' + x + '" y1="144" x2="' + x + '" y2="188"></line>';
    });
    return '<svg class="ecn-topo-svg" viewBox="0 0 560 230" role="img" aria-label="leaf-spine topology">' +
      links + spines + leaves + hostLinks + hosts +
      '<text class="ecn-topo-label" x="10" y="14">leaf-spine</text></svg>';
  }

  function railSvg() {
    let svg = '<svg class="ecn-topo-svg" viewBox="0 0 560 230" role="img" aria-label="rail-optimized topology">';
    const rails = [
      { x: 60, label: "rail 0" },
      { x: 320, label: "rail 1" },
    ];
    rails.forEach((r) => {
      svg += '<text class="ecn-topo-label" x="' + r.x + '" y="14">' + r.label + "</text>";
      svg += node(r.x + 30, 24, 110, 26, "spine", "spine");
      const leaves = [r.x - 10, r.x + 100];
      leaves.forEach((lx, i) => {
        svg += '<line class="ecn-topo-link" x1="' + (r.x + 85) + '" y1="52" x2="' + (lx + 40) +
          '" y2="118"></line>';
        svg += node(lx, 118, 80, 26, "leaf " + (i + 1), "leaf");
      });
      const hosts = [r.x + 5, r.x + 115];
      hosts.forEach((hx, i) => {
        svg += '<line class="ecn-topo-link dim" x1="' + (hx + 34) + '" y1="144" x2="' + (hx + 34) +
          '" y2="188"></line>';
        svg += node(hx, 188, 68, 24, "NIC " + (i + 1), "host");
      });
    });
    svg += '<text class="ecn-topo-label" x="150" y="226">rail 之间不交叉；每 rail 独立无损域</text>';
    return svg + "</svg>";
  }

  function node(x, y, w, h, label, cls) {
    return '<rect class="ecn-topo-node ' + cls + '" x="' + x + '" y="' + y + '" width="' + w +
      '" height="' + h + '" rx="5"></rect><text class="ecn-topo-node-label" x="' + (x + w / 2) +
      '" y="' + (y + h / 2 + 4) + '" text-anchor="middle">' + esc(label) + "</text>";
  }

  /* ------------------------------------------------------------- helpers */

  function sectionHead(sec) {
    return '<h2 class="sec-title">' + esc(sec.title) + "</h2>" +
      '<p class="page-lead">' + esc(sec.lead) + "</p>";
  }

  function ctl(label, input) {
    return '<label class="sim-field"><span>' + esc(label) + "</span>" + input + "</label>";
  }

  function stat(label, value, sub) {
    return '<div class="stat"><div class="stat-label">' + esc(label) + '</div><div class="stat-value">' +
      value + '</div><div class="stat-sub">' + esc(sub) + "</div></div>";
  }

  function details(q, a) {
    return '<details class="thinkbox"><summary>' + esc(q) + "</summary><p>" + esc(a) + "</p></details>";
  }

  function arr(x) {
    return Array.isArray(x) ? x : [];
  }

  function esc(s) {
    if (s === null || s === undefined) return "";
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
})();
