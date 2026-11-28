/* Document chapter: what a GPU is, how it differs from a CPU, and where the
 * GPU / NPU / DCU software stacks fit. Content lives in web/tools/gpu.json;
 * example/gpu_sim.py is injected by build.py as data.simCode + data.simOutput.
 *
 * Interactives (state local to this IIFE, listeners bound to the current host,
 * ids prefixed gpu-):
 *   1. CPU vs GPU metric bars (metric x accelerator)
 *   2. unified storage hierarchy ladder (CPU cache vs GPU cache)
 *   3. task-placement selector (CPU or GPU should own this task?)
 *   4. cache-line / coalescing path (contiguous vs strided warp load)
 */
(function () {
  const LK = (window.LK = window.LK || {});
  const G = (LK.gpu = LK.gpu || {});

  // Order-of-magnitude numbers, same as example/gpu_sim.py.
  const CPUS = {
    desktop: { id: "desktop", name: "CPU 桌面 16c", tflops: 4, bw: 80, lanes: 256, cache: 32 },
    server: { id: "server", name: "CPU 服务器 32c", tflops: 8, bw: 400, lanes: 512, cache: 128 },
  };
  const GPUS = [
    { id: "h100", name: "H100 SXM", tflops: 990, bw: 3350, lanes: 16896, cache: 50, mem: "HBM3 80 GB" },
    { id: "a100", name: "A100 SXM", tflops: 312, bw: 2039, lanes: 13824, cache: 40, mem: "HBM2e 80 GB" },
    { id: "rtx4090", name: "RTX 4090", tflops: 165, bw: 1008, lanes: 16384, cache: 72, mem: "GDDR6X 24 GB" },
    { id: "ascend910b", name: "Ascend 910B", tflops: 376, bw: 1600, lanes: null, cache: null, mem: "HBM2e 64 GB" },
    { id: "hcu", name: "HCU（海光 DCU）", tflops: 200, bw: 1000, lanes: null, cache: null, mem: "HBM ~32 GB" },
  ];
  const METRICS = {
    tflops: {
      label: "FP16/BF16 TFLOPS（dense）", unit: "TFLOPS", cpu: (c) => c.tflops, get: (g) => g.tflops,
      note: "GPU 用 Tensor Core + 更多 lane；CPU 峰值按 AVX512-BF16，桌面约 4、服务器约 8 TFLOPS。",
    },
    bw: {
      label: "内存带宽", unit: "GB/s", cpu: (c) => c.bw, get: (g) => g.bw,
      note: "桌面 DDR5 双通道 ~80 GB/s、服务器 8 通道 ~400 GB/s vs HBM3 3.35 TB/s；decode 是 memory-bound。",
    },
    lanes: {
      label: "核数 / FP32 lane 数", unit: "lanes", cpu: (c) => c.lanes, get: (g) => g.lanes,
      note: "CPU 桌面 16×16=256、服务器 32×16=512；GPU 132 SM×128=16896；NPU/DCU 的 lane 计数口径不同。",
    },
    cache: {
      label: "L3 / L2 缓存容量", unit: "MiB", cpu: (c) => c.cache, get: (g) => g.cache,
      note: "CPU 桌面 L3 32 MiB、服务器 128 MiB；GPU 的 L2 带宽更高、服务高并发访问；容量不是全部。",
    },
  };

  const CACHE_LINE = 128;
  const SECTOR = 32;

  const cmpState = { metric: "tflops", gpu: "h100", cpu: "server" };
  const hierState = { side: "gpu", idx: 2, hl: false };
  const scenState = { idx: 2 };
  const pathState = { mode: 0 };

  let timer = null;

  G.stop = function () {
    if (timer) clearInterval(timer);
    timer = null;
  };

  G.render = function (el, data, world) {
    G.stop();
    if (!el) return;
    data = data || {};
    el.innerHTML =
      '<p class="kicker">' + esc(data.kicker) + "</p>" +
      '<h1 class="page-title">' + esc(data.title) + "</h1>" +
      '<p class="page-lead">' + esc(data.lead) + "</p>" +
      whyHTML(data.why) +
      compareHTML(data.compare) +
      llmHTML(data.llm) +
      replaceHTML(data.replace) +
      memoryHTML(data.memory) +
      cacheHTML(data.cache) +
      accessHTML(data.access) +
      ecoHTML(data.eco) +
      codeHTML(data.codeSection, data.simOutput) +
      checkHTML(data.check);

    mountCompare(el.querySelector("#gpu-compare"));
    mountHierarchy(el.querySelector("#gpu-hierarchy"), data.memory, data.cache);
    mountScenario(el.querySelector("#gpu-scenario"), data.replace);
    mountPath(el.querySelector("#gpu-cache-path"), data.cache);

    const codeEl = el.querySelector("#gpu-sim-code");
    if (codeEl && LK.codeview && data.simCode) LK.codeview.mount(codeEl, data.simCode.lines);
  };

  /* ------------------------------------------------------------- sections */

  function whyHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="net-cards">' +
      arr(sec.roles).map((r, i) =>
        '<div class="net-card"><span class="train-no">0' + (i + 1) + "</span><b>" + esc(r.name) +
        '</b><span class="gpu-role-tag">' + esc(r.tag) + "</span><p>" + esc(r.does) +
        '</p><span class="net-when">' + esc(r.why) + "</span></div>"
      ).join("") + "</div>" +
      '<h3 class="sec-sub">本章要回答的四个问题</h3>' +
      '<ol class="gpu-questions">' + arr(sec.questions).map((q) => "<li>" + esc(q) + "</li>").join("") + "</ol>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function compareHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="gpu-goals">' +
      arr(sec.goals).map((g) =>
        '<div class="gpu-goal"><b>' + esc(g.name) + "</b><p>" + esc(g.detail) + "</p></div>"
      ).join("") + "</div>" +
      '<div class="table-wrap"><table class="tagtable"><thead><tr>' +
      "<th>维度</th><th>CPU（延迟优化）</th><th>GPU（吞吐优化）</th><th>为什么不一样</th>" +
      "</tr></thead><tbody>" +
      arr(sec.rows).map((r) =>
        "<tr><td><b>" + esc(r.item) + "</b></td><td>" + esc(r.cpu) + "</td><td>" + esc(r.gpu) +
        "</td><td>" + esc(r.why) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<h3 class="sec-sub">' + esc(sec.calcTitle) + "</h3>" +
      '<div id="gpu-compare" class="gpu-compare"></div>' +
      '<p class="note">' + esc(sec.calcNote) + "</p>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function llmHTML(sec) {
    sec = sec || {};
    const roof = sec.roofline || {};
    return sectionHead(sec) +
      '<div class="gpu-two-col">' +
      '<div class="gpu-panel"><h3 class="sec-sub">LLM 为什么适合 GPU</h3><ul class="gpu-list">' +
      arr(sec.whyGpu).map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul></div>" +
      '<div class="gpu-panel"><h3 class="sec-sub">CPU 为什么跟不上</h3><ul class="gpu-list">' +
      arr(sec.whyCpu).map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul></div>" +
      "</div>" +
      '<p class="note">账本假设：' + esc(roof.assumption) + "</p>" +
      '<div class="gpu-formula"><b>roofline 拐点</b><p>' + esc(roof.ridge) + "</p></div>" +
      '<div class="table-wrap"><table class="tagtable"><thead><tr>' +
      "<th>workload</th><th>FLOPs</th><th>字节</th><th>CPU 桌面</th><th>CPU 服务器</th><th>GPU</th>" +
      "<th>差距 桌/服</th><th>bound</th>" +
      "</tr></thead><tbody>" +
      arr(roof.rows).map((r) =>
        "<tr><td><b>" + esc(r.workload) + "</b></td><td>" + esc(r.flops) + "</td><td>" + esc(r.bytes) +
        "</td><td>" + esc(r.cpuDesktop) + "</td><td>" + esc(r.cpuServer) + "</td><td>" + esc(r.gpu) +
        "</td><td>" + esc(r.ratioDesktop) + " / " + esc(r.ratioServer) +
        "</td><td>" + esc(r.bound) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function replaceHTML(sec) {
    sec = sec || {};
    const amd = sec.amdahl || {};
    return sectionHead(sec) +
      '<div class="table-wrap"><table class="tagtable"><thead><tr>' +
      "<th>事项</th><th>CPU 能做 / 擅长</th><th>GPU 的限制</th>" +
      "</tr></thead><tbody>" +
      arr(sec.rows).map((r) =>
        "<tr><td><b>" + esc(r.item) + "</b></td><td>" + esc(r.cpu) + "</td><td>" + esc(r.gpu) + "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<div class="gpu-formula"><b>Amdahl 示例</b><p>' + esc(amd.example) + "</p></div>" +
      '<p class="note">' + esc(amd.lesson) + "</p>" +
      '<div class="gpu-callout"><b>结论</b><p>' + esc(sec.conclusion) + "</p></div>" +
      '<h3 class="sec-sub">' + esc(sec.scenarioTitle) + "</h3>" +
      '<div id="gpu-scenario" class="gpu-scenario-host"></div>' +
      '<p class="note">' + esc(sec.scenarioNote) + "</p>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function memoryHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<h3 class="sec-sub">' + esc(sec.calcTitle) + "</h3>" +
      '<div id="gpu-hierarchy" class="gpu-hierarchy-host"></div>' +
      '<p class="note">' + esc(sec.calcNote) + "</p>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function cacheHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="gpu-callout"><b>直接回答</b><p>' + esc(sec.answer) + "</p></div>" +
      '<ul class="gpu-list gpu-facts">' +
      arr(sec.facts).map((f) => "<li>" + esc(f) + "</li>").join("") + "</ul>" +
      '<h3 class="sec-sub">' + esc(sec.pathTitle) + "</h3>" +
      '<div id="gpu-cache-path" class="gpu-path-host"></div>' +
      '<p class="note">' + esc(sec.pathNote) + "</p>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function accessHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<ul class="gpu-list">' + arr(sec.rules).map((r) => "<li>" + esc(r) + "</li>").join("") + "</ul>" +
      '<h3 class="sec-sub">和 CPU cache 优化有什么不同</h3>' +
      '<ul class="gpu-list gpu-facts">' +
      arr(sec.compareCpu).map((r) => "<li>" + esc(r) + "</li>").join("") + "</ul>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function ecoHTML(sec) {
    sec = sec || {};
    return sectionHead(sec) +
      '<div class="table-wrap"><table class="tagtable gpu-eco-table"><thead><tr>' +
      "<th>软件栈</th><th>厂商</th><th>是什么</th><th>芯片</th><th>推理框架 / 迁移</th><th>迁移成本</th>" +
      "</tr></thead><tbody>" +
      arr(sec.rows).map((r) =>
        "<tr><td><b>" + esc(r.stack) + "</b></td><td>" + esc(r.vendor) + "</td><td>" + esc(r.kind) +
        "</td><td>" + esc(r.chips) + "</td><td>" + esc(r.frameworks) + "</td><td>" + esc(r.migrate) +
        "</td></tr>"
      ).join("") + "</tbody></table></div>" +
      '<ul class="gpu-list gpu-facts">' +
      arr(sec.notes).map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul>" +
      '<p class="note">' + esc(sec.note) + "</p>";
  }

  function codeHTML(sec, simOutput) {
    sec = sec || {};
    return sectionHead(sec) +
      '<pre class="train-run"><code>' + esc(sec.run) + "</code></pre>" +
      '<ul class="train-rules">' + arr(sec.notes).map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul>" +
      '<div class="train-code-head"><span>' + esc(sec.file) + "</span><span>纯 Python · 可运行</span></div>" +
      '<div class="train-code" id="gpu-sim-code"></div>' +
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

  /* ------------------------------------------------------ 1. compare bars */

  function mountCompare(host) {
    if (!host) return;
    host.innerHTML =
      '<div class="gpu-controls">' +
      ctl("指标", '<select data-cmp="metric">' +
        Object.keys(METRICS).map((k) => option(k, METRICS[k].label, cmpState.metric)).join("") +
        "</select>") +
      ctl("CPU 档位", '<select data-cmp="cpu">' +
        Object.values(CPUS).map((c) => option(c.id, c.name, cmpState.cpu)).join("") +
        "</select>") +
      ctl("GPU 档位", '<select data-cmp="gpu">' +
        GPUS.map((g) => option(g.id, g.name, cmpState.gpu)).join("") +
        "</select>") +
      "</div>" +
      '<div id="gpu-compare-body"></div>';
    host.querySelectorAll("[data-cmp]").forEach((sel) => {
      sel.addEventListener("change", () => {
        const f = sel.dataset.cmp;
        if (f === "metric") cmpState.metric = sel.value;
        else if (f === "cpu") cmpState.cpu = sel.value;
        else cmpState.gpu = sel.value;
        updateCompare(host);
      });
    });
    updateCompare(host);
  }

  function updateCompare(host) {
    const body = host.querySelector("#gpu-compare-body");
    if (!body) return;
    const metric = METRICS[cmpState.metric] || METRICS.tflops;
    const gpu = GPUS.filter((g) => g.id === cmpState.gpu)[0] || GPUS[0];
    const cpu = CPUS[cmpState.cpu] || CPUS.server;
    const cpuV = metric.cpu(cpu);
    const gpuV = metric.get(gpu);
    const max = Math.max(cpuV, gpuV === null || gpuV === undefined ? 0 : gpuV);
    const bar = (label, value, cls) => {
      if (value === null || value === undefined) return "";
      const width = Math.max(1.5, (value / max) * 100).toFixed(1);
      return '<div class="gpu-bar-row"><span>' + esc(label) + '</span><div class="gpu-bar-track"><i class="' +
        cls + '" style="width:' + width + '%"></i></div><b>' + fmtValue(value, metric) + "</b></div>";
    };
    let ratioText = "口径不同 / 未公开";
    if (gpuV !== null && gpuV !== undefined) {
      if (gpuV >= cpuV) ratioText = (gpuV / cpuV).toFixed(1) + "×（GPU 更高）";
      else ratioText = (cpuV / gpuV).toFixed(1) + "×（CPU 更高）";
    }
    body.innerHTML =
      '<div class="compare">' +
      stat("CPU", fmtValue(cpuV, metric), cpu.name + " · 量级值") +
      stat("GPU", gpuV === null || gpuV === undefined ? "n/a" : fmtValue(gpuV, metric), gpu.name + " · " + gpu.mem) +
      stat("差距", ratioText, metric.label) +
      "</div>" +
      '<div class="gpu-bars">' + bar("CPU", cpuV, "cpu") + bar(gpu.name, gpuV, "gpu") + "</div>" +
      '<p class="note">' + esc(metric.note) + "</p>" +
      (gpuV === null || gpuV === undefined
        ? '<p class="note">昇腾/HCU 的 lane 与缓存口径和 GPU 不同（AI Core / Unified Buffer），这里不强行换算。</p>'
        : "");
  }

  function fmtValue(value, metric) {
    if (value === null || value === undefined) return "n/a";
    if (metric.unit === "lanes") return fmt(value) + " lanes";
    return fmt(value) + " " + metric.unit;
  }

  /* ------------------------------------------------- 2. hierarchy ladder */

  function mountHierarchy(host, memory, cache) {
    if (!host) return;
    memory = memory || {};
    const gpuLevels = arr(memory.gpuLevels);
    const cpuLevels = arr(memory.cpuLevels);
    host.innerHTML =
      '<div class="gpu-hier">' +
      '<div class="gpu-hier-col"><div class="gpu-hier-head">GPU（device）</div><div class="gpu-hier-list">' +
      gpuLevels.map((l, i) => layerBtn("gpu", i, l)).join("") + "</div></div>" +
      '<div class="gpu-hier-col"><div class="gpu-hier-head">CPU（host）</div><div class="gpu-hier-list">' +
      cpuLevels.map((l, i) => layerBtn("cpu", i, l)).join("") + "</div></div>" +
      "</div>" +
      '<div class="gpu-hier-actions">' +
      '<button type="button" class="gpu-hit" data-hier-hl="1" aria-pressed="false">GPU 有 L1/L2 吗？高亮对应层</button>' +
      "</div>" +
      '<div id="gpu-hier-detail"></div>';
    host.querySelectorAll("[data-hier-side]").forEach((b) => {
      b.addEventListener("click", () => {
        hierState.side = b.dataset.hierSide;
        hierState.idx = +b.dataset.hierIdx;
        hierState.hl = false;
        updateHier(host, memory, cache);
      });
    });
    const hl = host.querySelector("[data-hier-hl]");
    if (hl) {
      hl.addEventListener("click", () => {
        hierState.hl = !hierState.hl;
        hl.setAttribute("aria-pressed", hierState.hl ? "true" : "false");
        updateHier(host, memory, cache);
      });
    }
    updateHier(host, memory, cache);
  }

  function layerBtn(side, idx, l) {
    return '<button type="button" class="gpu-layer" data-hier-side="' + side + '" data-hier-idx="' + idx +
      '" data-hier-level="' + esc(l.level) + '" aria-pressed="false"><b>' + esc(l.level) +
      "</b><em>" + esc(l.cap) + "</em></button>";
  }

  function updateHier(host, memory, cache) {
    memory = memory || {};
    const gpuLevels = arr(memory.gpuLevels);
    const cpuLevels = arr(memory.cpuLevels);
    host.querySelectorAll("[data-hier-side]").forEach((b) => {
      const on = b.dataset.hierSide === hierState.side && +b.dataset.hierIdx === hierState.idx;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
      const isGpuCache = b.dataset.hierSide === "gpu" &&
        (String(b.dataset.hierLevel).indexOf("L1") === 0 || String(b.dataset.hierLevel).indexOf("L2") === 0);
      b.classList.toggle("hl", hierState.hl && isGpuCache);
    });
    const body = host.querySelector("#gpu-hier-detail");
    if (!body) return;
    const answer = (cache && cache.answer) || "GPU 有 L1/L2，但 L1 与 shared 统一、L2 芯片级共享，没有 CPU 式 LLC/目录一致性。";
    if (hierState.hl) {
      const l1 = gpuLevels.filter((l) => String(l.level).indexOf("L1") === 0)[0];
      const l2 = gpuLevels.filter((l) => String(l.level).indexOf("L2") === 0)[0];
      body.innerHTML =
        '<div class="gpu-callout"><b>直接回答</b><p>' + esc(answer) + "</p></div>" +
        detailHTML("GPU", l1) + detailHTML("GPU", l2);
    } else {
      const list = hierState.side === "gpu" ? gpuLevels : cpuLevels;
      body.innerHTML = detailHTML(hierState.side === "gpu" ? "GPU" : "CPU", list[hierState.idx]);
    }
  }

  function detailHTML(side, l) {
    if (!l) return "";
    return '<div class="gpu-detail">' +
      '<div class="gpu-detail-head"><span class="gpu-side">' + esc(side) + "</span><b>" + esc(l.level) +
      "</b><em>" + esc(l.cache) + "</em></div>" +
      '<div class="gpu-detail-grid">' +
      detailItem("容量", l.cap) + detailItem("延迟", l.lat) +
      detailItem("带宽", l.bw) + detailItem("作用", l.role) +
      "</div></div>";
  }

  function detailItem(label, value) {
    return '<div class="gpu-detail-item"><span>' + esc(label) + "</span><b>" + esc(value) + "</b></div>";
  }

  /* ------------------------------------------------------ 3. task owner */

  function mountScenario(host, replace) {
    if (!host) return;
    replace = replace || {};
    const scenarios = arr(replace.scenarios);
    if (!scenarios.length) return;
    host.innerHTML =
      '<div class="gpu-scenarios">' +
      scenarios.map((s, i) =>
        '<button type="button" class="gpu-scenario' + (i === scenState.idx ? " on" : "") +
        '" data-scen="' + i + '"><b>' + esc(s.name) + '</b><span class="gpu-owner ' +
        (String(s.owner).toUpperCase() === "CPU" ? "cpu" : "gpu") + '">' + esc(s.owner) + "</span></button>"
      ).join("") + "</div>" +
      '<div id="gpu-scen-body"></div>';
    host.querySelectorAll("[data-scen]").forEach((b) => {
      b.addEventListener("click", () => {
        scenState.idx = +b.dataset.scen;
        host.querySelectorAll("[data-scen]").forEach((x) =>
          x.classList.toggle("on", +x.dataset.scen === scenState.idx)
        );
        updateScenario(host, replace);
      });
    });
    updateScenario(host, replace);
  }

  function updateScenario(host, replace) {
    const body = host.querySelector("#gpu-scen-body");
    if (!body) return;
    const scenarios = arr((replace || {}).scenarios);
    const s = scenarios[scenState.idx] || scenarios[0];
    if (!s) {
      body.innerHTML = "";
      return;
    }
    const isCpu = String(s.owner).toUpperCase() === "CPU";
    body.innerHTML =
      '<div class="gpu-scen-result">' +
      '<span class="gpu-owner ' + (isCpu ? "cpu" : "gpu") + '">' + esc(s.owner) + " 主导</span>" +
      "<p>" + esc(s.why) + "</p>" +
      "</div>" +
      '<p class="note">' + (isCpu
        ? "这类任务留在 CPU：GPU 的 launch、分支发散或 IO 限制会让它更慢。"
        : "这类任务交给 GPU：拆成大量规则并行工作，才能吃满 lane 和 HBM 带宽。") + "</p>";
  }

  /* ------------------------------------------------- 4. cache path demo */

  function pathCalc(stride) {
    const sectorOf = (t) => Math.floor((t * stride * 4) / SECTOR);
    const lineOf = (t) => Math.floor((t * stride * 4) / CACHE_LINE);
    const secSeen = {};
    const lineSeen = {};
    let sectors = 0;
    let lines = 0;
    for (let t = 0; t < 32; t++) {
      const s = sectorOf(t);
      const l = lineOf(t);
      if (!secSeen[s]) {
        secSeen[s] = true;
        sectors += 1;
      }
      if (!lineSeen[l]) {
        lineSeen[l] = true;
        lines += 1;
      }
    }
    const useful = 32 * 4;
    const moved = sectors * SECTOR;
    return { stride, sectorOf, lineOf, sectors, lines, moved, useful, util: useful / moved };
  }

  function mountPath(host, cache) {
    if (!host) return;
    cache = cache || {};
    const cases = arr(cache.path);
    if (!cases.length) return;
    host.innerHTML =
      '<div class="gpu-toggle">' +
      cases.map((c, i) =>
        '<button type="button" data-path="' + i + '" class="' + (i === pathState.mode ? "on" : "") + '">' +
        esc(c.case) + "</button>"
      ).join("") + "</div>" +
      '<div id="gpu-path-body"></div>';
    host.querySelectorAll("[data-path]").forEach((b) => {
      b.addEventListener("click", () => {
        pathState.mode = +b.dataset.path;
        host.querySelectorAll("[data-path]").forEach((x) =>
          x.classList.toggle("on", +x.dataset.path === pathState.mode)
        );
        updatePath(host, cache);
      });
    });
    updatePath(host, cache);
  }

  function updatePath(host, cache) {
    const body = host.querySelector("#gpu-path-body");
    if (!body) return;
    const cases = arr((cache || {}).path);
    const c = cases[pathState.mode] || cases[0];
    const stride = pathState.mode === 0 ? 1 : 8;
    const r = pathCalc(stride);
    const utilPct = r.util * 100;
    const tone = r.util >= 0.999 ? "ok" : r.util >= 0.5 ? "warn" : "bad";
    let cells = "";
    for (let t = 0; t < 32; t++) {
      const addr = t * stride * 4;
      const line = r.lineOf(t);
      cells += '<div class="gpu-sector gpu-sec-' + (line % 6) + '" title="thread ' + t + " → " + addr +
        " B，line " + line + '"><b>t' + t + "</b><em>" + addr + " B</em></div>";
    }
    body.innerHTML =
      '<div class="compare">' +
      stat("sector 数", String(r.sectors), "32 B sector，一次 warp load") +
      stat("触及 cache line", String(r.lines), CACHE_LINE + " B line") +
      stat("HBM 搬运", r.moved + " B", "sectors × 32 B") +
      stat("有效带宽", utilPct.toFixed(1) + "%", "128 B 有用 / 实际搬运") +
      "</div>" +
      '<div class="gpu-sector-grid">' + cells + "</div>" +
      '<div class="gpu-util"><div class="gpu-util-label"><span>line 利用率</span><span>' +
      utilPct.toFixed(1) + '%</span></div><div class="gpu-util-bar"><i class="' + tone +
      '" style="width:' + Math.max(1, Math.min(100, utilPct)).toFixed(1) + '%"></i></div></div>' +
      '<p class="note">' + esc(c.steps) + "</p>" +
      '<p class="note"><b>' + esc(c.verdict) + "</b></p>";
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

  function esc(s) {
    if (s === null || s === undefined) return "";
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
})();
