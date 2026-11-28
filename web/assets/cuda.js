/* Document chapter: CUDA execution model.
 * Content lives in web/tools/cuda.json; example/cuda_sim.py is injected by
 * build.py as data.simCode + data.simOutput.
 *
 * Interactives (all state local to this IIFE, listeners bound to the current
 * host element, ids prefixed cuda-):
 *   1. occupancy calculator   (block size x regs/thread x shared/block)
 *   2. coalescing stride picker (32 threads, 32 B sectors)
 *   3. roofline batch slider   (decode point moves, prefill point fixed)
 *   4. CUDA Graph launch comparison (layers x kernels per layer)
 */
(function () {
  const LK = (window.LK = window.LK || {});
  const C = (LK.cuda = LK.cuda || {});

  // H100-class device, order of magnitude (same constants as example/cuda_sim.py).
  const GPU = {
    sms: 132,
    maxWarps: 64,
    maxBlocks: 32,
    regs: 65536,
    smem: 228 * 1024,
  };
  // 30B-class MoE, same model constants as example/net_sim.py.
  const MODEL = {
    activeParams: 3.35e9,
    layers: 48,
    kvPerToken: 96 * 1024,
    hbmGbps: 3350,
    bf16Tflops: 990,
  };
  const RIDGE = (MODEL.bf16Tflops * 1e12) / (MODEL.hbmGbps * 1e9); // 295.5 FLOP/B
  const SECTOR_BYTES = 32;
  const SEQ = 4096;

  const occState = { block: 256, regs: 32, smemKb: 16 };
  const coalState = { stride: 1 };
  const roofState = { batch: 1 };
  const graphState = { layers: 48, perLayer: 8 };

  let timer = null;

  C.stop = function () {
    if (timer) clearInterval(timer);
    timer = null;
  };

  C.render = function (el, data, world) {
    C.stop();
    if (!el) return;
    data = data || {};
    el.innerHTML =
      '<p class="kicker">' + esc(data.kicker) + "</p>" +
      '<h1 class="page-title">' + esc(data.title) + "</h1>" +
      '<p class="page-lead">' + esc(data.lead) + "</p>" +
      whyHTML(data.why) +
      execHTML(data.exec) +
      memoryHTML(data.memory) +
      coalesceHTML(data.coalesce) +
      rooflineHTML(data.roofline) +
      graphHTML(data.graph) +
      kvHTML(data.kv) +
      codeHTML(data.codeSection, data.simOutput) +
      checkHTML(data.check);

    mountOcc(el.querySelector("#cuda-occ"));
    mountCoalesce(el.querySelector("#cuda-coalesce"));
    mountRoofline(el.querySelector("#cuda-roofline"));
    mountGraph(el.querySelector("#cuda-graph"));

    const codeEl = el.querySelector("#cuda-sim-code");
    if (codeEl && LK.codeview && data.simCode) LK.codeview.mount(codeEl, data.simCode.lines);
  };

  /* ------------------------------------------------------------- sections */

  function whyHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="net-cards">' +
      arr(sec.items).map((x, i) =>
        '<div class="net-card"><span class="train-no">0' + (i + 1) + "</span><b>" + esc(x.name) +
        "</b><p>" + esc(x.data) + '</p><span class="net-when">' + esc(x.when) + "</span></div>"
      ).join("") + "</div>";
  }

  function execHTML(sec) {
    sec = sec || {};
    const occ = sec.occupancy || {};
    return sectionHead(sec) +
      '<div class="table-wrap"><table class="tagtable net-table"><thead><tr>' +
      "<th>层 / 单位</th><th>是什么</th><th>执行模型里的作用</th>" +
      "</tr></thead><tbody>" +
      arr(sec.rows).map((r) =>
        "<tr><td><b>" + esc(r.term) + "</b></td><td>" + esc(r.what) + "</td><td>" + esc(r.why) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<h3 class="sec-sub">occupancy：SM 里到底能同时放多少</h3>' +
      '<p class="note">' + esc(occ.definition) + "</p>" +
      '<ul class="train-rules">' + arr(occ.limits).map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul>" +
      '<div class="table-wrap"><table class="tagtable net-table"><thead><tr>' +
      "<th>block</th><th>warps/block</th><th>常驻 block/SM</th><th>active warps</th><th>occupancy</th><th>谁先卡住</th>" +
      "</tr></thead><tbody>" +
      arr(occ.table).map((r) =>
        "<tr><td><b>" + esc(r.block) + "</b></td><td>" + esc(r.warps) + "</td><td>" + esc(r.resident) +
        "</td><td>" + esc(r.activeWarps) + "</td><td>" + esc(r.occupancy) + "</td><td>" + esc(r.binding) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<p class="note">' + esc(occ.note) + "</p>" +
      '<h3 class="sec-sub">' + esc(sec.calcTitle) + "</h3>" +
      '<div id="cuda-occ" class="cuda-occ"></div>' +
      '<p class="note">' + esc(sec.calcNote) + "</p>";
  }

  function memoryHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="table-wrap"><table class="tagtable net-table"><thead><tr>' +
      "<th>层级</th><th>容量</th><th>延迟</th><th>带宽</th><th>为什么关心</th>" +
      "</tr></thead><tbody>" +
      arr(sec.levels).map((l) =>
        "<tr><td><b>" + esc(l.level) + "</b></td><td>" + esc(l.capacity) + "</td><td>" + esc(l.latency) +
        "</td><td>" + esc(l.bandwidth) + "</td><td>" + esc(l.why) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<ul class="train-rules">' + arr(sec.rules).map((r) => "<li>" + esc(r) + "</li>").join("") + "</ul>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function coalesceHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="table-wrap"><table class="tagtable net-table"><thead><tr>' +
      "<th>stride（元素）</th><th>sectors / warp</th><th>实际搬运</th><th>有效字节</th><th>有效带宽</th>" +
      "</tr></thead><tbody>" +
      arr(sec.table).map((r) =>
        "<tr><td><b>" + esc(r.stride) + "</b></td><td>" + esc(r.sectors) + "</td><td>" + esc(r.moved) +
        "</td><td>" + esc(r.useful) + "</td><td>" + esc(r.util) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<ul class="train-rules">' + arr(sec.notes).map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul>" +
      '<h3 class="sec-sub">' + esc(sec.calcTitle) + "</h3>" +
      '<div id="cuda-coalesce" class="cuda-coalesce"></div>' +
      '<p class="note">' + esc(sec.calcNote) + "</p>";
  }

  function rooflineHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="cuda-formula"><b>拐点 / ridge point</b><p>' + esc(sec.ridge) + "</p></div>" +
      '<div class="table-wrap"><table class="tagtable net-table"><thead><tr>' +
      "<th>workload</th><th>M</th><th>arithmetic intensity</th><th>bound</th><th>t_mem</th><th>t_compute</th>" +
      "</tr></thead><tbody>" +
      arr(sec.table).map((r) =>
        "<tr><td><b>" + esc(r.workload) + "</b></td><td>" + esc(r.m) + "</td><td>" + esc(r.ai) +
        "</td><td>" + esc(r.bound) + "</td><td>" + esc(r.tMem) + "</td><td>" + esc(r.tCompute) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<ul class="train-rules">' + arr(sec.points).map((p) => "<li>" + esc(p) + "</li>").join("") + "</ul>" +
      '<h3 class="sec-sub">' + esc(sec.calcTitle) + "</h3>" +
      '<div id="cuda-roofline" class="cuda-roofline"></div>' +
      '<p class="note">' + esc(sec.note) + "</p>" +
      '<p class="note">' + esc(sec.calcNote) + "</p>";
  }

  function graphHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<ul class="train-rules">' + arr(sec.facts).map((f) => "<li>" + esc(f) + "</li>").join("") + "</ul>" +
      '<h3 class="sec-sub">' + esc(sec.calcTitle) + "</h3>" +
      '<div id="cuda-graph" class="cuda-graph"></div>' +
      '<ul class="train-rules">' + arr(sec.notes).map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul>" +
      '<p class="note">' + esc(sec.calcNote) + "</p>";
  }

  function kvHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="cuda-formula"><b>decode 字节账</b><p>' + esc(sec.formula) + "</p></div>" +
      '<div class="table-wrap"><table class="tagtable net-table"><thead><tr>' +
      "<th>场景</th><th>字节账</th><th>结果</th>" +
      "</tr></thead><tbody>" +
      arr(sec.exampleRows).map((r) =>
        "<tr><td><b>" + esc(r.label) + "</b></td><td>" + esc(r.formula) + "</td><td>" + esc(r.result) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<h3 class="sec-sub">优化手段 → 减少什么 → 谁负责</h3>' +
      '<div class="table-wrap"><table class="tagtable net-table"><thead><tr>' +
      "<th>优化手段</th><th>减少什么</th><th>谁负责（LLM / GPU / 引擎）</th>" +
      "</tr></thead><tbody>" +
      arr(sec.table).map((r) =>
        "<tr><td><b>" + esc(r.opt) + "</b></td><td>" + esc(r.saves) + "</td><td>" + esc(r.owner) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<ul class="train-rules">' + arr(sec.notes).map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul>";
  }

  function codeHTML(sec, simOutput) {
    sec = sec || {};
    return sectionHead(sec) +
      '<pre class="train-run"><code>' + esc(sec.run) + "</code></pre>" +
      '<ul class="train-rules">' + arr(sec.notes).map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul>" +
      '<div class="train-code-head"><span>' + esc(sec.file) + "</span><span>纯 Python · 可运行</span></div>" +
      '<div class="train-code" id="cuda-sim-code"></div>' +
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

  /* ---------------------------------------------------------- 1. occupancy */

  function occCalc(block, regsPerThread, smemKb) {
    const warps = Math.ceil(block / 32);
    const regsPerWarp = Math.ceil((regsPerThread * 32) / 256) * 256;
    const regsPerBlock = regsPerWarp * warps;
    const smemBytes = smemKb * 1024;
    const limits = {
      regs: Math.floor(GPU.regs / regsPerBlock),
      smem: smemBytes ? Math.floor(GPU.smem / smemBytes) : GPU.maxBlocks,
      blocks: GPU.maxBlocks,
      warps: Math.floor(GPU.maxWarps / warps),
    };
    const order = ["regs", "smem", "blocks", "warps"];
    let binding = order[0];
    order.forEach((k) => {
      if (limits[k] < limits[binding]) binding = k;
    });
    const min = limits[binding];
    const ties = order.filter((k) => limits[k] === min);
    const resident = Math.max(0, min);
    const activeWarps = resident * warps;
    const occupancy = Math.min(1, activeWarps / GPU.maxWarps);
    return { warps, regsPerBlock, limits, binding, ties, resident, activeWarps, occupancy };
  }

  const BINDING_LABEL = { regs: "寄存器", smem: "shared memory", blocks: "block 数", warps: "warp 数" };

  function mountOcc(host) {
    if (!host) return;
    host.innerHTML =
      '<div class="cuda-controls">' +
      ctl("block size", '<select data-occ="block">' +
        [32, 64, 128, 256, 512, 1024].map((v) => option(v, v + " threads", occState.block)).join("") +
        "</select>") +
      ctl("寄存器 / 线程", '<select data-occ="regs">' +
        [16, 24, 32, 40, 48, 64].map((v) => option(v, v + " regs", occState.regs)).join("") +
        "</select>") +
      ctl("shared / block", '<select data-occ="smem">' +
        [0, 8, 16, 32, 48, 100].map((v) => option(v, v + " KiB", occState.smemKb)).join("") +
        "</select>") +
      "</div>" +
      '<div id="cuda-occ-body"></div>';
    host.querySelectorAll("[data-occ]").forEach((sel) => {
      sel.addEventListener("change", () => {
        const f = sel.dataset.occ;
        if (f === "block") occState.block = +sel.value;
        else if (f === "regs") occState.regs = +sel.value;
        else occState.smemKb = +sel.value;
        updateOcc(host);
      });
    });
    updateOcc(host);
  }

  function updateOcc(host) {
    const body = host.querySelector("#cuda-occ-body");
    if (!body) return;
    const r = occCalc(occState.block, occState.regs, occState.smemKb);
    const pct = r.occupancy * 100;
    const tone = pct >= 99.9 ? "ok" : pct >= 50 ? "warn" : "bad";
    const bindLabel = r.ties.map((k) => BINDING_LABEL[k]).join(" / ");
    const bindingRow =
      '<div class="table-wrap"><table class="tagtable cuda-limit-table"><thead><tr>' +
      "<th>约束</th><th>本配置用量</th><th>SM 上限</th><th>还能放几个 block</th>" +
      "</tr></thead><tbody>" +
      limitRow(r, "regs", "寄存器", fmt(r.regsPerBlock) + " regs/block", fmt(GPU.regs) + " regs") +
      limitRow(r, "smem", "shared memory", occState.smemKb ? occState.smemKb + " KiB/block" : "0（不占）", "228 KiB") +
      limitRow(r, "blocks", "block 数", "1 block", "32 blocks") +
      limitRow(r, "warps", "warp 数", r.warps + " warps/block", "64 warps") +
      "</tbody></table></div>";

    body.innerHTML =
      '<div class="compare">' +
      stat("常驻 block / SM", String(r.resident), "受 " + bindLabel + " 限制") +
      stat("active warps", r.activeWarps + " / 64", r.warps + " warps × " + r.resident + " block") +
      stat("occupancy", pct.toFixed(1) + "%", "active warps / 64 warp slot") +
      stat("谁先卡住", bindLabel + (pct >= 99.9 ? "（已打满）" : ""), "换参数时观察这一项") +
      "</div>" +
      '<div class="cuda-meter"><div class="cuda-meter-bar"><i class="' + tone + '" style="width:' +
      Math.max(1, Math.min(100, pct)).toFixed(1) + '%"></i></div><span>' + pct.toFixed(1) + "% occupancy</span></div>" +
      bindingRow +
      '<p class="note">' + occNote(r) + "</p>";
  }

  function limitRow(r, key, label, use, cap) {
    const on = r.ties.indexOf(key) !== -1 ? " class=\"on\"" : "";
    return '<tr' + on + "><td>" + esc(label) + "</td><td>" + esc(use) + "</td><td>" + esc(cap) +
      "</td><td>" + (r.limits[key] === undefined ? "—" : r.limits[key]) + "</td></tr>";
  }

  function occNote(r) {
    const slots = GPU.maxWarps - r.activeWarps;
    if (r.activeWarps >= GPU.maxWarps) {
      return "occupancy 已打满（64/64 warps）：当前 " + r.warps + " warps/block × " + r.resident +
        " block = 64 warps，" + r.ties.map((k) => BINDING_LABEL[k]).join(" / ") +
        " 同时到上限。打满只说明有足够 warp 掩盖延迟，不代表一定更快；还要看 bank conflict 和实际字节数。";
    }
    let why;
    if (r.binding === "regs") {
      why = "寄存器先卡住：每个 block 要 " + fmt(r.regsPerBlock) +
        " 个寄存器，65536 只够 " + r.limits.regs + " 个 block；降低 regs/thread 或换更小的 block。";
    } else if (r.binding === "smem") {
      why = "shared memory 先卡住：每个 block 声明 " + occState.smemKb + " KiB，228 KiB 只够 " +
        r.limits.smem + " 个 block；减少 shared/block，或增大 block 让 warp 数占满。";
    } else if (r.binding === "warps") {
      why = "warp 数先卡住：每个 block " + r.warps + " 个 warp，64 个 warp slot 只够 " +
        r.limits.warps + " 个 block。";
    } else {
      why = "32 blocks/SM 上限先卡住：block 再小也不会多，SM 里最多 32 个 block。";
    }
    return why + " 当前 " + r.activeWarps + "/64 warps，还有 " + slots +
      " 个 slot 空着；occupancy 低不一定慢，但要看有没有足够 warp 掩盖 HBM 延迟。";
  }

  /* --------------------------------------------------------- 2. coalescing */

  function coalCalc(stride) {
    const sectorOf = (t) => Math.floor((t * stride * 4) / SECTOR_BYTES);
    const seen = {};
    let sectors = 0;
    for (let t = 0; t < 32; t++) {
      const s = sectorOf(t);
      if (!seen[s]) {
        seen[s] = true;
        sectors += 1;
      }
    }
    const useful = 32 * 4;
    const moved = sectors * SECTOR_BYTES;
    return { stride, sectorOf, sectors, useful, moved, util: useful / moved };
  }

  function mountCoalesce(host) {
    if (!host) return;
    host.innerHTML =
      '<div class="cuda-controls cuda-controls-1">' +
      ctl("stride（元素）", '<select data-coal="stride">' +
        [1, 2, 4, 8, 32].map((v) => option(v, "stride = " + v, coalState.stride)).join("") +
        "</select>") +
      "</div>" +
      '<div id="cuda-coal-body"></div>';
    host.querySelectorAll("[data-coal]").forEach((sel) => {
      sel.addEventListener("change", () => {
        coalState.stride = +sel.value;
        updateCoal(host);
      });
    });
    updateCoal(host);
  }

  function updateCoal(host) {
    const body = host.querySelector("#cuda-coal-body");
    if (!body) return;
    const c = coalCalc(coalState.stride);
    const utilPct = c.util * 100;
    const tone = c.util >= 0.999 ? "ok" : c.util >= 0.5 ? "warn" : "bad";
    let cells = "";
    for (let t = 0; t < 32; t++) {
      const addr = t * c.stride * 4;
      const sector = c.sectorOf(t);
      cells += '<div class="cuda-sector cuda-sec-' + (sector % 6) + '" title="thread ' + t +
        " → " + addr + " B，sector " + sector + '"><b>t' + t + "</b><em>" + addr + " B</em></div>";
    }
    body.innerHTML =
      '<p class="cuda-legend">32 个线程各读 4 B；格子顶边颜色 = 该线程命中的 32 B sector（同色同 sector）。</p>' +
      '<div class="cuda-sector-grid">' + cells + "</div>" +
      '<div class="compare">' +
      stat("sectors / warp", String(c.sectors), "一次 warp load 触及的 32 B sector 数") +
      stat("实际搬运", c.moved + " B", "sectors × 32 B") +
      stat("有效字节", c.useful + " B", "32 × 4 B") +
      stat("有效带宽", utilPct.toFixed(1) + "%", "有效字节 / 实际搬运") +
      "</div>" +
      '<div class="cuda-util"><div class="cuda-util-label"><span>sector 利用率</span><span>' +
      utilPct.toFixed(1) + '%</span></div><div class="cuda-util-bar"><i class="' + tone +
      '" style="width:' + Math.max(1, Math.min(100, utilPct)).toFixed(1) + '%"></i></div></div>' +
      '<p class="note">' + coalNote(c) + "</p>";
  }

  function coalNote(c) {
    if (c.stride === 1) {
      return "连续 128 B = 4 个 sector，一次 transaction 搬完，有效带宽 100%。这是 coalesced 访问的理想情况。";
    }
    if (c.stride === 2) {
      return "线程间隔 8 B，覆盖 256 B = 8 个 sector；只用到 128 B，一半 HBM 带宽被浪费。";
    }
    if (c.stride === 4) {
      return "覆盖 512 B = 16 个 sector，有效带宽 25%；访问模式越稀疏，搬回来的无用字节越多。";
    }
    return "每个线程独占一个 32 B sector，最多 32 个 sector：搬 1024 B 只用到 128 B，8× 放大，有效带宽 12.5%。" +
      " paged attention 的 block_table gather 就是这种形状。";
  }

  /* ----------------------------------------------------------- 3. roofline */

  function roofCalc(batch) {
    const weightBytes = MODEL.activeParams * 2;
    const kvBytes = batch * SEQ * MODEL.kvPerToken;
    const flops = 2 * MODEL.activeParams * batch;
    const bytes = weightBytes + kvBytes;
    const ai = flops / bytes;
    const bound = ai >= RIDGE ? "compute" : "memory";
    const tMemMs = (bytes / (MODEL.hbmGbps * 1e9)) * 1e3;
    const tCompMs = (flops / (MODEL.bf16Tflops * 1e12)) * 1e3;
    const roofTf = Math.min(MODEL.bf16Tflops, (ai * MODEL.hbmGbps) / 1000);
    return { batch, weightBytes, kvBytes, flops, bytes, ai, bound, tMemMs, tCompMs, roofTf };
  }

  function mountRoofline(host) {
    if (!host) return;
    host.innerHTML =
      '<div class="cuda-controls cuda-controls-1">' +
      '<label class="sim-field"><span>decode batch <b id="cuda-roof-batch">' + roofState.batch +
      '</b></span><input type="range" data-roof="batch" min="1" max="128" step="1" value="' +
      roofState.batch + '"></label>' +
      "</div>" +
      '<div id="cuda-roof-body"></div>';
    host.querySelectorAll("[data-roof]").forEach((inp) => {
      inp.addEventListener("input", () => {
        roofState.batch = +inp.value;
        const label = host.querySelector("#cuda-roof-batch");
        if (label) label.textContent = roofState.batch;
        updateRoof(host);
      });
    });
    updateRoof(host);
  }

  function updateRoof(host) {
    const body = host.querySelector("#cuda-roof-body");
    if (!body) return;
    const r = roofCalc(roofState.batch);
    const boundText = r.bound === "memory" ? "memory（带宽斜线）" : "compute（算力平顶）";
    const tone = r.bound === "memory" ? "warn" : "ok";
    body.innerHTML =
      '<div class="cuda-roof">' + roofSvg(r) + "</div>" +
      '<div class="compare">' +
      stat("arithmetic intensity", fmtAi(r.ai) + " FLOP/B", "2·N·batch / (2·N + batch·seq·KV/token)") +
      stat("所在 bound", '<span class="cuda-bound ' + tone + '">' + boundText + "</span>",
        "拐点 ridge = " + RIDGE.toFixed(1) + " FLOP/B") +
      stat("每步字节", (r.bytes / 1e9).toFixed(2) + " GB",
        "权重 " + (r.weightBytes / 1e9).toFixed(2) + " GB + KV " + (r.kvBytes / 1e9).toFixed(2) + " GB") +
      stat("t_mem / t_compute", r.tMemMs.toFixed(2) + " / " + r.tCompMs.toFixed(2) + " ms",
        r.tMemMs > r.tCompMs ? "时间由带宽决定" : "时间由算力决定") +
      "</div>" +
      '<p class="note">' + roofNote(r) + "</p>";
  }

  function roofSvg(r) {
    const W = 560, H = 280, left = 48, right = 544, top = 14, bottom = 244;
    const xmin = 0.5, xmax = 10000, ymin = 0.5, ymax = 2000;
    const lg = Math.log10;
    const xOf = (v) => left + ((lg(v) - lg(xmin)) / (lg(xmax) - lg(xmin))) * (right - left);
    const yOf = (v) => bottom - ((lg(v) - lg(ymin)) / (lg(ymax) - lg(ymin))) * (bottom - top);
    let grid = "";
    [1, 10, 100, 1000, 10000].forEach((t) => {
      const x = xOf(t);
      grid += '<line class="cuda-roof-grid" x1="' + x.toFixed(1) + '" y1="' + top + '" x2="' + x.toFixed(1) +
        '" y2="' + bottom + '"></line><text class="cuda-roof-label" x="' + x.toFixed(1) + '" y="' + (bottom + 15) +
        '" text-anchor="middle">' + t + "</text>";
    });
    [1, 10, 100, 1000].forEach((t) => {
      const y = yOf(t);
      grid += '<line class="cuda-roof-grid" x1="' + left + '" y1="' + y.toFixed(1) + '" x2="' + right +
        '" y2="' + y.toFixed(1) + '"></line><text class="cuda-roof-label" x="' + (left - 6) + '" y="' +
        (y + 4).toFixed(1) + '" text-anchor="end">' + t + "</text>";
    });
    const peakY = yOf(MODEL.bf16Tflops);
    const ridgeX = xOf(RIDGE);
    const roofLine = '<polyline class="cuda-roof-line" points="' +
      xOf(xmin).toFixed(1) + "," + yOf(xmin * MODEL.hbmGbps / 1000).toFixed(1) + " " +
      ridgeX.toFixed(1) + "," + peakY.toFixed(1) + " " +
      xOf(xmax).toFixed(1) + "," + peakY.toFixed(1) + '"></polyline>';
    const ridge = '<line class="cuda-roof-ridge" x1="' + ridgeX.toFixed(1) + '" y1="' + top +
      '" x2="' + ridgeX.toFixed(1) + '" y2="' + peakY.toFixed(1) + '"></line>' +
      '<text class="cuda-roof-label ridge" x="' + (ridgeX - 5).toFixed(1) + '" y="' + (top + 12) +
      '" text-anchor="end">ridge ' + RIDGE.toFixed(1) + "</text>";
    const prefillX = xOf(3863.8);
    const prefill = '<circle class="cuda-roof-point prefill" cx="' + prefillX.toFixed(1) + '" cy="' +
      peakY.toFixed(1) + '" r="4"></circle><text class="cuda-roof-label prefill" x="' +
      (prefillX - 7).toFixed(1) + '" y="' + (peakY - 8).toFixed(1) +
      '" text-anchor="end">prefill M=4096</text>';
    const decX = xOf(r.ai);
    const decY = yOf(r.roofTf);
    const decode = '<circle class="cuda-roof-point decode" cx="' + decX.toFixed(1) + '" cy="' +
      decY.toFixed(1) + '" r="5"></circle><text class="cuda-roof-label decode" x="' +
      (decX + 8).toFixed(1) + '" y="' + (decY - 8).toFixed(1) +
      '">decode b=' + r.batch + "</text>";
    return '<svg class="cuda-roof-svg" viewBox="0 0 ' + W + " " + H +
      '" role="img" aria-label="roofline: decode batch ' + r.batch + " at arithmetic intensity " +
      r.ai.toFixed(2) + ' FLOP per byte">' +
      grid + roofLine + ridge + prefill + decode +
      '<text class="cuda-roof-label axis" x="' + (right) + '" y="' + (H - 6) +
      '" text-anchor="end">arithmetic intensity（FLOP/byte，log）</text>' +
      '<text class="cuda-roof-label axis" x="8" y="14">TFLOPS</text>' +
      "</svg>";
  }

  function roofNote(r) {
    const saturated = (2 * MODEL.activeParams) / (SEQ * MODEL.kvPerToken);
    if (r.batch === 1) {
      return "batch=1：AI ≈ " + r.ai.toFixed(2) + "，远低于拐点 " + RIDGE.toFixed(1) +
        "，落在带宽斜线上；时间由 " + (r.bytes / 1e9).toFixed(2) + " GB / 3.35 TB/s 决定，" +
        "算力只用到约 " + r.roofTf.toFixed(2) + " TFLOPS。";
    }
    return "batch=" + r.batch + "：权重被批量摊薄，但每请求 KV 线性增加；AI 升到 " + fmtAi(r.ai) +
      "，仍低于拐点。继续加 batch，AI 会饱和在 2N/(seq·KV/token) ≈ " + saturated.toFixed(1) +
      " FLOP/B，所以 decode 始终是 memory-bound：" + r.tMemMs.toFixed(2) + " ms 带宽 vs " +
      r.tCompMs.toFixed(2) + " ms 算力。";
  }

  /* -------------------------------------------------------- 4. CUDA Graph */

  function graphCalc(layers, perLayer) {
    const launchUs = 4;
    const kernelUs = 3;
    const graphUs = 6;
    const kernels = layers * perLayer;
    const eagerCpu = kernels * launchUs;
    const gpuBusy = kernels * kernelUs;
    const eagerStep = Math.max(eagerCpu, gpuBusy);
    const graphStep = graphUs + gpuBusy;
    const saving = 1 - graphUs / eagerCpu;
    const speedup = eagerStep / graphStep;
    return { kernels, eagerCpu, gpuBusy, eagerStep, graphCpu: graphUs, graphStep, saving, speedup };
  }

  function mountGraph(host) {
    if (!host) return;
    host.innerHTML =
      '<div class="cuda-controls cuda-controls-2">' +
      ctl("LLM 层数", '<select data-graph="layers">' +
        [16, 24, 32, 48, 64].map((v) => option(v, v + " 层", graphState.layers)).join("") +
        "</select>") +
      ctl("每层 kernel 数", '<select data-graph="perLayer">' +
        [4, 6, 8, 12].map((v) => option(v, v + " 个", graphState.perLayer)).join("") +
        "</select>") +
      "</div>" +
      '<div id="cuda-graph-body"></div>';
    host.querySelectorAll("[data-graph]").forEach((sel) => {
      sel.addEventListener("change", () => {
        const f = sel.dataset.graph;
        if (f === "layers") graphState.layers = +sel.value;
        else graphState.perLayer = +sel.value;
        updateGraph(host);
      });
    });
    updateGraph(host);
  }

  function updateGraph(host) {
    const body = host.querySelector("#cuda-graph-body");
    if (!body) return;
    const g = graphCalc(graphState.layers, graphState.perLayer);
    const max = Math.max(g.eagerCpu, g.gpuBusy, g.graphCpu);
    const w = (v) => Math.max(1.5, (v / max) * 100).toFixed(1) + "%";
    const bar = (label, value, cls) =>
      '<div class="cuda-bar-row"><span>' + esc(label) + '</span><div class="cuda-bar-track"><i class="' +
      cls + '" style="width:' + w(value) + '"></i></div><b>' + value.toFixed(0) + " µs</b></div>";
    body.innerHTML =
      '<div class="compare">' +
      stat("kernel 数 / 步", String(g.kernels), graphState.layers + " 层 × " + graphState.perLayer + " 个") +
      stat("eager CPU 开销", g.eagerCpu.toFixed(0) + " µs", g.kernels + " 次 launch × 4 µs") +
      stat("graph replay CPU", g.graphCpu.toFixed(0) + " µs", "一次 cudaGraphLaunch 提交整张图") +
      stat("CPU 开销节省", (g.saving * 100).toFixed(1) + "%", "只省 launch，不省 HBM 字节") +
      "</div>" +
      '<div class="cuda-bars">' +
      bar("eager CPU launch", g.eagerCpu, "eager") +
      bar("GPU 实际忙", g.gpuBusy, "gpu") +
      bar("graph replay CPU", g.graphCpu, "graph") +
      "</div>" +
      '<div class="compare">' +
      stat("step（eager → graph）", g.eagerStep.toFixed(0) + " → " + g.graphStep.toFixed(0) + " µs",
        "eager 受 CPU 限制，graph 受 GPU 限制") +
      stat("step 加速", g.speedup.toFixed(2) + "x", "Amdahl：省掉的是 CPU 空转") +
      "</div>" +
      '<p class="note">GPU 只忙 ' + g.gpuBusy.toFixed(0) + " µs，eager 下 CPU 要 " + g.eagerCpu.toFixed(0) +
      " µs，中间约 " + (g.eagerStep - g.gpuBusy).toFixed(0) +
      " µs 是 GPU 等 CPU；graph 把这段空转消掉后，step 接近 GPU 下界，" +
      "但 kernel 计算和 HBM 字节一个都没少。</p>";
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

  function fmt(n) {
    return Number(n).toLocaleString("en-US");
  }

  function fmtAi(ai) {
    return ai >= 100 ? ai.toFixed(1) : ai.toFixed(2);
  }

  function esc(s) {
    if (s === null || s === undefined) return "";
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
})();
