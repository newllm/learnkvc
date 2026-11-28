/* Document chapter: LLM optimization — FlashAttention, PagedAttention,
 * quantization, speculative decoding, scheduling, model structure and a
 * workload -> bottleneck -> technique decision guide.
 *
 * Content lives in web/tools/opt.json; example/opt_sim.py is injected by
 * build.py as data.simCode + data.simOutput.
 *
 * Interactives (state local to this IIFE, listeners bound to the current host,
 * ids prefixed opt-):
 *   1. bottleneck -> recommended directions
 *   2. FlashAttention HBM traffic (N picker)
 *   3. quantization calculator (preset x weight bits x KV bits)
 *   4. speculative decoding speedup (alpha / draft cost / gamma)
 */
(function () {
  const LK = (window.LK = window.LK || {});
  const O = (LK.opt = LK.opt || {});

  // Fallbacks mirror example/opt_sim.py (量级).
  const ATTENTION = { d: 128, elemBytes: 2, ns: [1024, 4096, 16384, 65536] };
  const QUANT = { bwGbps: 3350, seq: 4096, batch: 1, kvPerTokenFp16: 57344 };
  const FALLBACK_PRESETS = [
    { id: "qwen30b", name: "Qwen3-Coder-30B-A3B", params: 30.5e9, kvPerToken: 98304 },
    { id: "qwen7b", name: "Qwen2.5-Coder-7B", params: 7.6e9, kvPerToken: 57344 },
    { id: "qwen32b", name: "Qwen2.5-Coder-32B", params: 32.8e9, kvPerToken: 262144 },
  ];
  const WEIGHT_BIT_OPTIONS = [16, 8, 4];
  const KV_BIT_OPTIONS = [16, 8];
  const GAMMA_OPTIONS = [2, 4, 8];

  const state = {
    bnIdx: 0,
    flashN: 4096,
    presetId: "qwen30b",
    weightBits: 16,
    kvBits: 16,
    alpha: 0.7,
    draftC: 0.1,
    gamma: 4,
  };

  let timer = null;

  O.stop = function () {
    if (timer) clearInterval(timer);
    timer = null;
  };

  O.render = function (el, data, world) {
    O.stop();
    if (!el) return;
    data = data || {};
    el.innerHTML =
      '<p class="kicker">' + esc(data.kicker) + "</p>" +
      '<h1 class="page-title">' + esc(data.title) + "</h1>" +
      '<p class="page-lead">' + esc(data.lead) + "</p>" +
      whyHTML(data.why) +
      mapHTML(data.map) +
      flashHTML(data.flash) +
      pagedHTML(data.paged) +
      quantHTML(data.quant) +
      decodeHTML(data.decode) +
      scheduleHTML(data.schedule) +
      modelHTML(data.model) +
      chooseHTML(data.choose) +
      codeHTML(data.codeSection, data.simOutput) +
      checkHTML(data.check);

    mountBottleneck(el.querySelector("#opt-bottleneck"), data.map);
    mountFlash(el.querySelector("#opt-flash"), data.flash);
    mountQuant(el.querySelector("#opt-quant"), data.quant, world);
    mountSpec(el.querySelector("#opt-spec"), data.decode);

    const codeEl = el.querySelector("#opt-sim-code");
    if (codeEl && LK.codeview && data.simCode) LK.codeview.mount(codeEl, data.simCode.lines);
  };

  /* ------------------------------------------------------------- sections */

  function whyHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="opt-constraints">' +
      arr(sec.constraints).map((c, i) =>
        '<div class="opt-constraint"><span class="opt-no">0' + (i + 1) + "</span><b>" + esc(c.name) +
        '</b><span class="opt-metric">' + esc(c.metric) + "</span><p>" + esc(c.detail) + "</p></div>"
      ).join("") + "</div>" +
      '<h3 class="sec-sub">和前面章节的关系</h3>' +
      '<ul class="opt-relation">' + arr(sec.relation).map((r) => "<li>" + esc(r) + "</li>").join("") + "</ul>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function mapHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="table-wrap"><table class="tagtable opt-cat-table"><thead><tr>' +
      "<th>类别</th><th>包含</th><th>攻击的瓶颈</th><th>代价</th><th>典型场景</th>" +
      "</tr></thead><tbody>" +
      arr(sec.categories).map((c) =>
        "<tr><td><b>" + esc(c.name) + "</b></td><td>" + esc(arr(c.items).join("、")) + "</td><td>" +
        esc(c.bottleneck) + "</td><td>" + esc(c.cost) + "</td><td>" + esc(c.example) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<h3 class="sec-sub">' + esc(sec.calcTitle) + "</h3>" +
      '<div id="opt-bottleneck" class="opt-bottleneck-host"></div>' +
      '<p class="note">' + esc(sec.calcNote) + "</p>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function flashHTML(sec) {
    sec = sec || {};
    const traffic = sec.traffic || {};
    return sectionHead(sec) +
      '<ul class="opt-idea">' + arr(sec.idea).map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul>" +
      '<h3 class="sec-sub">HBM 流量账本（单层 / 单 head）</h3>' +
      '<div class="table-wrap"><table class="tagtable opt-flash-table"><thead><tr>' +
      "<th>seq N</th><th>naive materialize</th><th>FlashAttention tiled</th><th>倍数</th>" +
      "</tr></thead><tbody>" +
      arr(traffic.rows).map((r) =>
        "<tr><td><b>" + esc(r.n) + "</b></td><td>" + esc(r.naive) + "</td><td>" + esc(r.tiled) +
        "</td><td>" + esc(r.ratio) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<p class="note">' + esc(traffic.note) + "</p>" +
      '<h3 class="sec-sub">prefill vs decode</h3>' +
      '<ul class="opt-relation">' + arr(sec.prefillDecode).map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul>" +
      '<h3 class="sec-sub">' + esc(sec.calcTitle) + "</h3>" +
      '<div id="opt-flash" class="opt-flash-host"></div>' +
      '<p class="note">' + esc(sec.calcNote) + "</p>" +
      '<h3 class="sec-sub">v1 / v2 / v3 一句话</h3>' +
      '<div class="opt-versions">' +
      arr(sec.versions).map((v) =>
        '<div class="opt-version"><b>' + esc(v.v) + "</b><p>" + esc(v.what) + "</p></div>"
      ).join("") + "</div>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function pagedHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="opt-points">' +
      arr(sec.points).map((p) =>
        '<div class="opt-point"><b>' + esc(p.name) + "</b><p>" + esc(p.detail) + "</p></div>"
      ).join("") + "</div>" +
      '<h3 class="sec-sub">和已有章节的关系</h3>' +
      '<ul class="opt-relation">' + arr(sec.relation).map((r) => "<li>" + esc(r) + "</li>").join("") + "</ul>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function quantHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="table-wrap"><table class="tagtable opt-quant-table"><thead><tr>' +
      "<th>方法</th><th>做法</th><th>精度口径</th><th>注意</th>" +
      "</tr></thead><tbody>" +
      arr(sec.methods).map((m) =>
        "<tr><td><b>" + esc(m.name) + "</b></td><td>" + esc(m.what) + "</td><td>" + esc(m.precision) +
        "</td><td>" + esc(m.note) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<h3 class="sec-sub">精度 vs 吞吐的取舍</h3>' +
      '<ul class="opt-relation">' + arr(sec.tradeoffs).map((t) => "<li>" + esc(t) + "</li>").join("") + "</ul>" +
      '<h3 class="sec-sub">' + esc(sec.calcTitle) + "</h3>" +
      '<div id="opt-quant" class="opt-quant-host"></div>' +
      '<p class="note">' + esc(sec.calcNote) + "</p>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function decodeHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<ul class="opt-idea">' + arr(sec.how).map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul>" +
      '<div class="opt-callout"><b>加速比公式</b><p>' + esc(sec.formula) + "</p></div>" +
      '<div class="table-wrap"><table class="tagtable opt-spec-table"><thead><tr>' +
      "<th>acceptance α</th>" + arr(sec.costs).map((c) => "<th>c=" + esc(c) + "</th>").join("") +
      "</tr></thead><tbody>" +
      arr(sec.table).map((r) =>
        "<tr><td><b>" + esc(r.alpha) + "</b></td>" +
        arr(r.values).map((v) => "<td>" + esc(v) + "</td>").join("") + "</tr>"
      ).join("") + "</tbody></table></div>" +
      '<h3 class="sec-sub">' + esc(sec.calcTitle) + "</h3>" +
      '<div id="opt-spec" class="opt-spec-host"></div>' +
      '<p class="note">' + esc(sec.calcNote) + "</p>" +
      '<div class="opt-callout"><b>树验证与 MTP</b><p>' + esc(sec.tree) + " " + esc(sec.mtp) + "</p></div>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function scheduleHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="table-wrap"><table class="tagtable opt-schedule-table"><thead><tr>' +
      "<th>技术</th><th>攻击</th><th>怎么做</th><th>代价</th>" +
      "</tr></thead><tbody>" +
      arr(sec.rows).map((r) =>
        "<tr><td><b>" + esc(r.tech) + "</b></td><td>" + esc(r.attack) + "</td><td>" + esc(r.how) +
        "</td><td>" + esc(r.cost) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function modelHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="table-wrap"><table class="tagtable opt-model-table"><thead><tr>' +
      "<th>结构</th><th>省什么</th><th>损失 / 代价</th><th>需要什么支持</th>" +
      "</tr></thead><tbody>" +
      arr(sec.rows).map((r) =>
        "<tr><td><b>" + esc(r.name) + "</b></td><td>" + esc(r.saves) + "</td><td>" + esc(r.costs) +
        "</td><td>" + esc(r.needs) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function chooseHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="opt-tree">' +
      arr(sec.tree).map((t) =>
        '<div class="opt-tree-card"><b>' + esc(t.workload) + '</b><span class="opt-bound">' +
        esc(t.bottleneck) + '</span><p class="opt-first">先上：' + esc(arr(t.first).join(" · ")) +
        '</p><p class="opt-avoid">避免：' + esc(arr(t.avoid).join(" · ")) + "</p></div>"
      ).join("") + "</div>" +
      '<h3 class="sec-sub">选型原则</h3>' +
      '<ul class="opt-relation">' + arr(sec.principles).map((p) => "<li>" + esc(p) + "</li>").join("") + "</ul>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function codeHTML(sec, simOutput) {
    sec = sec || {};
    return sectionHead(sec) +
      '<pre class="train-run"><code>' + esc(sec.run) + "</code></pre>" +
      '<ul class="train-rules">' + arr(sec.notes).map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul>" +
      '<div class="train-code-head"><span>' + esc(sec.file) + "</span><span>纯 Python · 可运行</span></div>" +
      '<div class="train-code" id="opt-sim-code"></div>' +
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

  /* --------------------------------------------------- 1. bottleneck picker */

  function mountBottleneck(host, mapSec) {
    if (!host) return;
    mapSec = mapSec || {};
    const bns = arr(mapSec.bottlenecks);
    if (!bns.length) return;
    state.bnIdx = Math.min(state.bnIdx, bns.length - 1);
    host.innerHTML =
      '<div class="opt-bn-buttons">' +
      bns.map((b, i) =>
        '<button type="button" class="opt-bn-btn' + (i === state.bnIdx ? " on" : "") +
        '" data-bn="' + i + '">' + esc(b.name) + "</button>"
      ).join("") + "</div>" +
      '<div id="opt-bn-body"></div>';
    host.querySelectorAll("[data-bn]").forEach((b) => {
      b.addEventListener("click", () => {
        state.bnIdx = +b.dataset.bn;
        host.querySelectorAll("[data-bn]").forEach((x) =>
          x.classList.toggle("on", +x.dataset.bn === state.bnIdx)
        );
        updateBottleneck(host, mapSec);
      });
    });
    updateBottleneck(host, mapSec);
  }

  function updateBottleneck(host, mapSec) {
    const body = host.querySelector("#opt-bn-body");
    if (!body) return;
    const bns = arr((mapSec || {}).bottlenecks);
    const b = bns[state.bnIdx] || bns[0];
    if (!b) {
      body.innerHTML = "";
      return;
    }
    const chips = (items, cls) =>
      '<div class="opt-chips">' +
      arr(items).map((x) => '<span class="opt-chip ' + cls + '">' + esc(x) + "</span>").join("") + "</div>";
    body.innerHTML =
      '<div class="opt-bn-result">' +
      "<p>" + esc(b.detail) + "</p>" +
      '<div class="opt-bn-cols">' +
      '<div class="opt-bn-col"><b>推荐方向</b>' + chips(b.recommend, "good") + "</div>" +
      '<div class="opt-bn-col"><b>先别做</b>' + chips(b.avoid, "warn") + "</div>" +
      "</div></div>";
  }

  /* ------------------------------------------------ 2. FlashAttention calc */

  function mountFlash(host, flashSec) {
    if (!host) return;
    const cfg = flashCfg(flashSec);
    const ns = arr(cfg.ns).length ? arr(cfg.ns) : ATTENTION.ns;
    if (ns.indexOf(state.flashN) === -1) state.flashN = ns[Math.floor(ns.length / 2)];
    host.innerHTML =
      '<div class="opt-controls">' +
      ctl("序列长度 N", '<select data-flash="n">' +
        ns.map((n) => option(n, fmtInt(n) + " tokens", state.flashN)).join("") + "</select>") +
      "</div>" +
      '<div id="opt-flash-body"></div>';
    host.querySelectorAll("[data-flash]").forEach((sel) => {
      sel.addEventListener("change", () => {
        state.flashN = +sel.value;
        updateFlash(host, cfg);
      });
    });
    updateFlash(host, cfg);
  }

  function flashCfg(flashSec) {
    const sim = (flashSec && flashSec.sim) || {};
    return {
      d: typeof sim.d === "number" ? sim.d : ATTENTION.d,
      b: typeof sim.elemBytes === "number" ? sim.elemBytes : ATTENTION.elemBytes,
      ns: arr(sim.ns).length ? arr(sim.ns) : ATTENTION.ns,
    };
  }

  function flashCalc(n, d, b) {
    const tiled = 4 * n * d * b;
    const naive = tiled + 4 * n * n * b;
    return { n, tiled, naive, ratio: naive / tiled };
  }

  function updateFlash(host, cfg) {
    const body = host.querySelector("#opt-flash-body");
    if (!body) return;
    const r = flashCalc(state.flashN, cfg.d, cfg.b);
    const max = r.naive;
    const bar = (label, value, cls) =>
      '<div class="opt-bar-row"><span>' + esc(label) + '</span><div class="opt-bar-track"><i class="' +
      cls + '" style="width:' + Math.max(1, (value / max) * 100).toFixed(1) + '%"></i></div><b>' +
      fmtBytes(value) + "</b></div>";
    body.innerHTML =
      '<div class="compare">' +
      stat("naive materialize", fmtBytes(r.naive), "写/读 N×N 分数与概率矩阵") +
      stat("FlashAttention tiled", fmtBytes(r.tiled), "只搬 Q/K/V/O（理想 tiling）") +
      stat("倍数", r.ratio.toFixed(1) + "×", "N 越大差距越大") +
      stat("等效 HBM 时间", (r.tiled / (3350e9) * 1e6).toFixed(1) + " µs", "3.35 TB/s 下 tiled 的 IO 时间（量级）") +
      "</div>" +
      '<div class="opt-bars">' + bar("naive", r.naive, "naive") + bar("tiled", r.tiled, "tiled") + "</div>" +
      '<p class="note">' +
      (state.flashN >= 4096
        ? "prefill 大 N：naive 的 N×N 物化是主瓶颈，tiling 收益随 N 增长；FLOPs 仍然是 O(N²)。"
        : "N 较小：tiling 收益有限；decode q=1 时没有 N×N 可省，收益接近 1×，要靠 FlashDecoding/split-K 提并行度。") +
      "</p>";
  }

  /* --------------------------------------------------- 3. quantization calc */

  function presetList(quantSec, world) {
    const out = [];
    const src = (world && Array.isArray(world.presets)) ? world.presets : [];
    src.forEach((p) => {
      const m = p.math || {};
      const params = Number(m.total) || Number(p.params) || 0;
      const kv = Number(m.kv_bytes) || Number(p.kvPerToken) || 0;
      if (params > 0 && kv > 0) out.push({ id: p.id, name: p.name || p.id, params, kvPerToken: kv });
    });
    if (!out.length) {
      arr((quantSec || {}).calcPresets).forEach((p) => {
        if (Number(p.params) > 0) {
          out.push({ id: p.id, name: p.name || p.id, params: Number(p.params), kvPerToken: Number(p.kvPerToken) || 57344 });
        }
      });
    }
    if (!out.length) out.push(FALLBACK_PRESETS[0]);
    return out;
  }

  function quantCfg(quantSec) {
    const sim = (quantSec && quantSec.sim) || {};
    return {
      bw: (Number(sim.bwGbps) || 3350) * 1e9,
      seq: Number(sim.seq) || 4096,
      kvPerTokenFp16: Number(sim.kv_per_token_fp16) || 57344,
    };
  }

  function quantCalc(preset, weightBits, kvBits, cfg) {
    const weightBytes = preset.params * weightBits / 8;
    const kvBytes = cfg.seq * preset.kvPerToken * (kvBits / 16);
    const total = weightBytes + kvBytes;
    const ms = total / cfg.bw * 1e3;
    const baseTotal = preset.params * 2 + cfg.seq * preset.kvPerToken; // BF16 weights + FP16 KV
    const baseMs = baseTotal / cfg.bw * 1e3;
    return { weightBytes, kvBytes, total, ms, baseTotal, baseMs, speedup: baseMs / ms };
  }

  function mountQuant(host, quantSec, world) {
    if (!host) return;
    const presets = presetList(quantSec, world);
    if (!presets.some((p) => p.id === state.presetId)) state.presetId = presets[0].id;
    const cfg = quantCfg(quantSec);
    host.innerHTML =
      '<div class="opt-controls opt-controls-3">' +
      ctl("模型预设", '<select data-quant="preset">' +
        presets.map((p) => option(p.id, p.name, state.presetId)).join("") + "</select>") +
      ctl("权重 bits", '<select data-quant="wb">' +
        WEIGHT_BIT_OPTIONS.map((b) => option(b, b + " bit" + (b === 8 ? "（FP8/INT8）" : b === 4 ? "（INT4）" : ""), state.weightBits)).join("") +
        "</select>") +
      ctl("KV bits", '<select data-quant="kb">' +
        KV_BIT_OPTIONS.map((b) => option(b, b + " bit" + (b === 8 ? "（FP8/INT8）" : "（FP16）"), state.kvBits)).join("") +
        "</select>") +
      "</div>" +
      '<div id="opt-quant-body"></div>';
    host.querySelectorAll("[data-quant]").forEach((sel) => {
      sel.addEventListener("change", () => {
        const f = sel.dataset.quant;
        if (f === "preset") state.presetId = sel.value;
        else if (f === "wb") state.weightBits = +sel.value;
        else state.kvBits = +sel.value;
        updateQuant(host, quantSec, world);
      });
    });
    updateQuant(host, quantSec, world);
  }

  function updateQuant(host, quantSec, world) {
    const body = host.querySelector("#opt-quant-body");
    if (!body) return;
    const presets = presetList(quantSec, world);
    const preset = presets.filter((p) => p.id === state.presetId)[0] || presets[0];
    const cfg = quantCfg(quantSec);
    const r = quantCalc(preset, state.weightBits, state.kvBits, cfg);
    const pctW = (r.weightBytes / r.total) * 100;
    const pctK = 100 - pctW;
    const tone = r.speedup >= 2 ? "ok" : r.speedup >= 1.2 ? "warn" : "dim";
    const kernels = state.weightBits === 4
      ? "W4A16 需要 GPTQ/AWQ/Marlin 等 kernel；没有 kernel 只省显存不加速。"
      : state.weightBits === 8
        ? "FP8 需要 Hopper+；INT8 需要 Tensor Core INT8 kernel。"
        : "BF16 是基线；KV 位宽降低仍受 attention kernel 支持限制。";
    body.innerHTML =
      '<div class="compare">' +
      stat("权重显存", fmtGB(r.weightBytes), state.weightBits + " bit × " + fmtB(preset.params) + " 参数") +
      stat("KV @ seq " + cfg.seq, fmtBytes(r.kvBytes), state.kvBits + " bit；FP16=" + fmtKiB(preset.kvPerToken) + "/token") +
      stat("decode 每步字节", fmtGB(r.total), "batch=1，权重 + KV") +
      stat("decode 时间", r.ms.toFixed(2) + " ms", (cfg.bw / 1e9).toFixed(0) + " GB/s 带宽模型") +
      stat("相对 BF16 加速", '<span class="opt-tone ' + tone + '">' + r.speedup.toFixed(2) + "×</span>",
        "BF16 基线 " + r.baseMs.toFixed(2) + " ms") +
      "</div>" +
      '<div class="opt-qbar"><i class="w" style="width:' + pctW.toFixed(1) + '%"></i><i class="kv" style="width:' +
      pctK.toFixed(1) + '%"></i></div>' +
      '<p class="note">权重 ' + pctW.toFixed(0) + "% · KV " + pctK.toFixed(0) + "%；" + kernels +
      " 7B/4K 单请求以权重为主，长上下文/大 batch 时 KV 占比上升。" +
      (state.weightBits <= 8 && state.kvBits <= 8 ? " 当前组合接近 INT4+FP8 的 3.9× 量级。" : "") +
      "</p>";
  }

  /* -------------------------------------------------- 4. speculative decode */

  function specCalc(alpha, c, gamma) {
    const accepted = alpha >= 1 ? gamma + 1 : (1 - Math.pow(alpha, gamma + 1)) / (1 - alpha);
    const time = 1 + c * gamma;
    return { accepted, time, speedup: accepted / time };
  }

  function mountSpec(host, decodeSec) {
    if (!host) return;
    const sim = (decodeSec && decodeSec.sim) || {};
    if (GAMMA_OPTIONS.indexOf(state.gamma) === -1) state.gamma = GAMMA_OPTIONS[1];
    host.innerHTML =
      '<div class="opt-controls">' +
      ctl("acceptance α", '<input type="range" data-spec="alpha" min="0.3" max="0.95" step="0.01" value="' +
        state.alpha + '"><b data-spec-val="alpha">' + state.alpha.toFixed(2) + "</b>") +
      ctl("草稿成本 c", '<input type="range" data-spec="c" min="0.02" max="0.5" step="0.01" value="' +
        state.draftC + '"><b data-spec-val="c">' + state.draftC.toFixed(2) + "</b>") +
      ctl("草稿 token γ", '<select data-spec="gamma">' +
        GAMMA_OPTIONS.map((g) => option(g, "γ = " + g, state.gamma)).join("") + "</select>") +
      "</div>" +
      '<div id="opt-spec-body"></div>';
    host.querySelectorAll("[data-spec]").forEach((inp) => {
      const evt = inp.tagName === "SELECT" ? "change" : "input";
      inp.addEventListener(evt, () => {
        const f = inp.dataset.spec;
        if (f === "alpha") state.alpha = +inp.value;
        else if (f === "c") state.draftC = +inp.value;
        else state.gamma = +inp.value;
        const la = host.querySelector('[data-spec-val="alpha"]');
        const lc = host.querySelector('[data-spec-val="c"]');
        if (la) la.textContent = state.alpha.toFixed(2);
        if (lc) lc.textContent = state.draftC.toFixed(2);
        updateSpec(host);
      });
    });
    updateSpec(host);
  }

  function updateSpec(host) {
    const body = host.querySelector("#opt-spec-body");
    if (!body) return;
    const r = specCalc(state.alpha, state.draftC, state.gamma);
    const tone = r.speedup >= 1.2 ? "ok" : r.speedup >= 1 ? "warn" : "bad";
    const verdict = r.speedup >= 1.2 ? "值得做" : r.speedup >= 1 ? "勉强持平" : "不划算";
    body.innerHTML =
      '<div class="compare">' +
      stat("每步接受 token", r.accepted.toFixed(2), "γ 个草稿 + 1 个 bonus") +
      stat("每步时间（目标 pass 单位）", r.time.toFixed(2), "1 + c·γ") +
      stat("加速比", '<span class="opt-tone ' + tone + '">' + r.speedup.toFixed(2) + "×</span>",
        "(1−α^(γ+1)) / ((1−α)(1+cγ))") +
      stat("结论", verdict, verdict === "不划算" ? "草稿太贵/接受率太低，不如普通 decode" : "继续做 A/B 验证") +
      "</div>" +
      '<p class="note">' +
      (r.speedup < 1
        ? "当前参数下加速比 <1：接受率低或草稿成本高，投机解码会亏；提高 α、降低 c 或减小 γ 再试。"
        : "量级模型：忽略树验证、调度和 batch 摊薄；真实收益要做端到端 benchmark。") +
      "</p>";
  }

  /* ------------------------------------------------------------- helpers */

  function sectionHead(sec) {
    return '<h2 class="sec-title">' + esc(sec.title) + "</h2>" +
      '<p class="page-lead">' + esc(sec.lead) + "</p>";
  }

  function ctl(label, input) {
    return '<label class="sim-field"><span>' + esc(label) + "</span>" + input + "</label>";
  }

  function option(value, label, current) {
    return '<option value="' + esc(value) + '"' +
      (String(value) === String(current) ? " selected" : "") + ">" + esc(label) + "</option>";
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

  function fmtInt(n) {
    return Number(n).toLocaleString("en-US");
  }

  function fmtB(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    return fmtInt(n);
  }

  function fmtBytes(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + " MB";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + " KB";
    return Math.round(n) + " B";
  }

  function fmtGB(n) {
    return (n / 1e9).toFixed(2) + " GB";
  }

  function fmtKiB(n) {
    if (n >= 1024 && n % 1024 === 0) return (n / 1024).toFixed(0) + " KiB";
    return fmtBytes(n);
  }

  function esc(s) {
    if (s === null || s === undefined) return "";
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
})();
