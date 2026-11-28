/* App shell: thinking-first navigation + chapter stepper + playback. */
(function () {
  const LK = (window.LK = window.LK || {});
  const D = window.LK_DATA;
  const TR = window.LK_TRACES && window.LK_TRACES.kv;

  // learnkvc → whykvc：旧 localStorage 不删除，首次用原值填到新 key，
  // 避免已学进度/主题/导航状态因为改名丢失。
  (function migrateOldKeys() {
    const pairs = {
      "learnkvc.progress.v1": "whykvc.progress.v1",
      "learnkvc.theme": "whykvc.theme",
      "learnkvc.stepnav": "whykvc.stepnav",
      "learnkvc.sectionnav": "whykvc.sectionnav",
      "learnkvc.reflect.v1": "whykvc.reflect.v1",
    };
    try {
      Object.keys(pairs).forEach((oldKey) => {
        const oldValue = localStorage.getItem(oldKey);
        if (oldValue !== null && localStorage.getItem(pairs[oldKey]) === null) {
          localStorage.setItem(pairs[oldKey], oldValue);
        }
      });
    } catch (e) {
      /* file:// / privacy mode：忽略 */
    }
  })();

  if (!D || !TR) {
    document.body.innerHTML =
      '<p style="padding:2rem;color:#d7e2ee">数据缺失：先运行 <code>python3 web/tools/build.py</code>。</p>';
    return;
  }

  const kv = D.kv;
  const MAX_GEN = TR.generated.length;
  const VIEWS = ["think", "kv", "prefix", "paged", "transformer", "anatomy", "calc", "train", "net", "nccl", "cuda", "ecn", "gpu", "npu", "opt", "feedback"];
  const state = { view: "think", step: 0, genStep: MAX_GEN, playing: false, timer: null };

  const $ = (sel) => document.querySelector(sel);
  const el = {
    think: $("#view-think"),
    kv: $("#view-kv"),
    prefix: $("#view-prefix"),
    paged: $("#view-paged"),
    transformer: $("#view-transformer"),
    anatomy: $("#view-anatomy"),
    calc: $("#view-calc"),
    train: $("#view-train"),
    net: $("#view-net"),
    nccl: $("#view-nccl"),
    cuda: $("#view-cuda"),
    ecn: $("#view-ecn"),
    gpu: $("#view-gpu"),
    npu: $("#view-npu"),
    opt: $("#view-opt"),
    feedback: $("#view-feedback"),
    tabs: Array.from(document.querySelectorAll("#tabs [data-view]")),
    code: $("#code"),
    codeMeta: $("#code-meta"),
    prefixCode: $("#prefix-code"),
    prefixCodeMeta: $("#prefix-code-meta"),
    prefixDoc: $("#prefix-doc"),
    prefixViz: $("#prefix-viz"),
    prefixStepNext: $("#prefix-step-next"),
    pagedCode: $("#paged-code"),
    pagedCodeMeta: $("#paged-code-meta"),
    pagedDoc: $("#paged-doc"),
    pagedViz: $("#paged-viz"),
    pagedStepNext: $("#paged-step-next"),
    stepNavs: {
      kv: $("#stepnav-kv"),
      prefix: $("#stepnav-prefix"),
      paged: $("#stepnav-paged"),
      train: $("#stepnav-train"),
      net: $("#stepnav-net"),
      nccl: $("#stepnav-nccl"),
      cuda: $("#stepnav-cuda"),
      ecn: $("#stepnav-ecn"),
      transformer: $("#stepnav-transformer"),
      anatomy: $("#stepnav-anatomy"),
    gpu: $("#stepnav-gpu"),
      npu: $("#stepnav-npu"),
      opt: $("#stepnav-opt"),
    },
    doc: $("#doc-card"),
    viz: $("#viz"),
    kvStepNext: $("#kv-step-next"),
    dots: $("#dots"),
    counters: $("#counters"),
    prev: $("#prev"),
    next: $("#next"),
    quiz: $("#quiz"),
    anatomyBody: $("#anatomy-body"),
    anatomyMain: $("#anatomy-main"),
    calcBody: $("#calc"),
    trainBody: $("#train-body"),
    netBody: $("#net-body"),
    ncclBody: $("#nccl-body"),
    cudaBody: $("#cuda-body"),
    ecnBody: $("#ecn-body"),
    transformerBody: $("#transformer-body"),
    gpuBody: $("#gpu-body"),
    npuBody: $("#npu-body"),
    optBody: $("#opt-body"),
    trainMain: $("#train-main"),
    netMain: $("#net-main"),
    ncclMain: $("#nccl-main"),
    cudaMain: $("#cuda-main"),
    ecnMain: $("#ecn-main"),
    transformerMain: $("#transformer-main"),
    gpuMain: $("#gpu-main"),
    npuMain: $("#npu-main"),
    optMain: $("#opt-main"),
    feedbackBody: $("#feedback"),
    ovWhy: $("#ov-why"),
  };

  const navGroups = Array.from(document.querySelectorAll("#tabs [data-nav-group]")).map((root) => {
    const trigger = root.querySelector("[data-nav-trigger]");
    const label = root.querySelector("[data-nav-label]");
    return {
      trigger,
      menu: root.querySelector(".nav-menu"),
      label,
      base: label.textContent,
    };
  });

  let codeView = null;
  let prefixCode = null;
  let pagedCode = null;

  state.prefix = { step: 0, scenario: "shared", snap: 0, playing: false, timer: null };
  state.paged = { step: 0, snap: 0, sel: "seq-b", block: null, playing: false, timer: null };
  const CHAPTERS = ["kv", "prefix", "paged"];
  // 按路线图的完整地图顺序串起所有章节；最后一章读完去学习反馈。
  const LEARNING_ORDER = [
    "gpu", "npu", "cuda", "transformer", "anatomy", "train",
    "kv", "prefix", "paged", "net", "nccl", "ecn", "opt",
  ];
  const CHAPTER_LABELS = {
    transformer: "Transformer 与 LLM",
    anatomy: "LLM 结构",
    gpu: "认识 GPU",
    npu: "NPU 与国产算力",
    cuda: "CUDA 执行模型",
    kv: "单请求 KV cache",
    prefix: "Prefix cache",
    paged: "PagedAttention",
    net: "推理网络",
    nccl: "NCCL 学习章节",
    ecn: "ECN + DCQCN",
    train: "LLM 训练",
    opt: "LLM 优化",
  };
  const DOC_CHAPTERS = ["transformer", "anatomy", "net", "train", "nccl", "cuda", "ecn", "gpu", "npu", "opt"];
  const ALL_CHAPTERS = CHAPTERS.concat(DOC_CHAPTERS);
  const DOC_SECTIONS = {
    transformer: ["why", "attn", "arch", "timeline", "scaling", "win", "limits", "codeSection", "check"],
    anatomy: ["composition", "package", "layers", "pipeline", "gpuLoad", "kernels", "gpuFunctionMap", "runtime", "codeSection", "check"],
    net: ["why", "nccl", "sim", "roce", "pd", "codeSection", "check"],
    train: ["stepSection", "ledgerSection", "parallelSection", "scriptSection", "codeSection", "checkSection"],
    nccl: ["why", "collectives", "ring", "tree", "model", "topology", "measure", "codeSection", "check"],
    cuda: ["why", "exec", "memory", "coalesce", "roofline", "graph", "kv", "codeSection", "check"],
    ecn: ["why", "layers", "switch", "server", "lossless", "priority", "ecn", "tuning", "codeSection", "check"],
    gpu: ["why", "compare", "llm", "replace", "memory", "cache", "access", "eco", "codeSection", "check"],
    npu: ["why", "compare", "vendors", "stack", "llm", "select", "codeSection", "check"],
    opt: ["why", "map", "flash", "paged", "quant", "decode", "schedule", "model", "choose", "codeSection", "check"],
  };
  const DOC_RENDERERS = { transformer: "transformer", anatomy: "anatomy", net: "net", train: "train", nccl: "nccl", cuda: "cuda", ecn: "ecn", gpu: "gpu", npu: "npu", opt: "opt" };
  const DOC_MODULES = { transformer: () => LK.transformer, anatomy: () => LK.anatomy, net: () => LK.net, train: () => LK.train, nccl: () => LK.nccl, cuda: () => LK.cuda, ecn: () => LK.ecn, gpu: () => LK.gpu, npu: () => LK.npu, opt: () => LK.opt };
  const STAGE_NAMES = ["think", "explore", "reflect"];
  state.stages = {};
  state.docStep = { transformer: 0, anatomy: 0, net: 0, train: 0, nccl: 0, cuda: 0, ecn: 0, gpu: 0, npu: 0, opt: 0 };
  state.docLock = { transformer: 0, anatomy: 0, net: 0, train: 0, nccl: 0, cuda: 0, ecn: 0, gpu: 0, npu: 0, opt: 0 };
  state.navCollapsed = { kv: false, prefix: false, paged: false, transformer: false, anatomy: false, net: false, train: false, nccl: false, cuda: false, ecn: false, gpu: false, npu: false, opt: false };
  const world = { presets: D.anatomy.presets, gpus: D.anatomy.context.gpus };

  const esc = (s) =>
    String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const clamp = (n) => Math.max(0, Math.min(kv.steps.length - 1, n));

  /* ------------------------------------------------------ chapter render */

  function renderDoc(step, index) {
    const n = kv.steps.length;
    const think = step.think
      ? '<details class="thinkbox step-think"><summary>请回答：' + esc(step.think.q) +
        "</summary><p>" + esc(step.think.a) + "</p></details>"
      : "";
    el.doc.innerHTML =
      '<div class="doc-kicker">第 ' + (index + 1) + " / " + n + " 步 · " +
      esc(step.sectionTitle.replace(/`/g, "")) + "</div>" +
      "<h2>" + esc(step.title) + "</h2>" +
      '<p class="hint">' + esc(step.hint) + "</p>" +
      think +
      '<div class="doc-body">' + step.bodyHTML + "</div>" +
      '<div class="doc-foot">讲解：<a href="../' + kv.docFile + '" target="_blank" rel="noopener">' +
      kv.docFile + '</a> · 代码：<a href="../' + kv.codeFile + '" target="_blank" rel="noopener">' +
      kv.codeFile + "</a></div>";
  }

  function renderViz() {
    const step = kv.steps[state.step];
    LK.viz.render(el.viz, step, {
      trace: TR,
      genStep: state.genStep,
      playing: state.playing,
      setGenStep,
      togglePlay,
    });
  }

  function renderCounters() {
    const s = state.genStep ? TR.steps[state.genStep - 1] : null;
    if (!s) {
      el.counters.innerHTML = "生成进度 t = 0 / " + MAX_GEN + " · 计数器归零";
      return;
    }
    el.counters.innerHTML =
      "无 cache <b>" + s.plain_pos + "</b> 次位置 / <b>" + s.plain_pairs.toLocaleString() + "</b> 对" +
      '<span class="sep">|</span>有 cache <b>' + s.cached_pos + "</b> 次 / <b>" +
      s.cached_pairs.toLocaleString() + "</b> 对" +
      '<span class="sep">|</span>省 <b>' + s.wasted + "</b> 个位置";
  }

  function resetKvScroll(right) {
    const main = right && right.closest(".kv-main");
    if (main) main.scrollTop = 0;
    else if (right) right.scrollTop = 0;
  }

  function applyStepCode(view, container, step) {
    const ranges = (step && step.codeRanges) || [];
    const hasCode = ranges.length > 0;
    const chapter = container && container.closest(".chapter");
    if (chapter) chapter.classList.toggle("no-code", !hasCode);
    if (!view) return hasCode;
    view.showOnly(hasCode ? ranges : []);
    view.highlight(hasCode ? ranges : []);
    return hasCode;
  }

  function renderStep() {
    const step = kv.steps[state.step];
    if (LK.progress) LK.progress.recordStep(step.id);
    renderDoc(step, state.step);
    resetKvScroll(document.getElementById("right-pane"));
    if (codeView) {
      const hasCode = applyStepCode(codeView, el.code, step);
      if (hasCode) codeView.scrollToLine(step.codeRanges[0].start);
      el.codeMeta.textContent = hasCode
        ? "本节代码 " + step.codeRanges.map((r) => r.start + "–" + r.end).join(" · ") + " 行"
        : "本节无代码";
    }
    renderViz();
    renderCounters();
    renderStepNext("kv");
    updateDots();
    renderChapterNav("kv");
    if (window.LK && LK.progress) {
      const l = LK.progress.lesson("kv");
      if ((l.exploreMax || 0) < state.step + 1) LK.progress.saveLesson("kv", { exploreMax: state.step + 1 });
      renderChapterNav("kv");
    }
  }

  function setStep(i) {
    state.step = clamp(i);
    const view = el.kv;
    const main = view && view.querySelector(".kv-main");
    if (stageOf("kv") === "explore" && main && main.classList.contains("flow-active")) {
      scrollKvTo("kv", state.step);
    } else {
      renderStep();
      updateHash();
    }
  }

  function setGenStep(g) {
    state.genStep = Math.max(0, Math.min(MAX_GEN, g));
    renderViz();
    renderCounters();
  }

  function togglePlay() {
    if (state.playing) {
      stopPlay();
      return;
    }
    if (state.genStep >= MAX_GEN) state.genStep = 0;
    state.playing = true;
    renderViz();
    renderCounters();
    state.timer = setInterval(() => {
      if (state.genStep >= MAX_GEN) {
        stopPlay();
        return;
      }
      state.genStep += 1;
      renderViz();
      renderCounters();
      if (state.genStep >= MAX_GEN) stopPlay();
    }, 750);
  }

  function stopPlay(rerender = true) {
    state.playing = false;
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
    if (rerender) {
      renderViz();
      renderCounters();
    }
  }

  /* ------------------------------------------------------- prefix chapter */

  function renderPrefixCounters() {
    const tr = LK_TRACES.prefix.scenarios[state.prefix.scenario];
    el.counters.innerHTML =
      "命中率 <b>" + Math.round(tr.stats.hit_rate * 100) + "%</b>" +
      '<span class="sep">|</span>复用 <b>' + tr.stats.reused_tokens + "</b> / " +
      tr.stats.prompt_tokens + " tok" +
      '<span class="sep">|</span>淘汰 <b>' + tr.stats.evicted + "</b> 块" +
      '<span class="sep">|</span>池 <b>' + tr.stats.peak_blocks + "</b> / " + tr.capacity;
  }

  function renderPrefixViz() {
    const step = LK_DATA.prefix.steps[state.prefix.step];
    if (step.viz === "reuse") {
      LK.workshop.renderReuse(el.prefixViz, LK_TRACES.kv.reuse, { compact: true });
      return;
    }
    LK.vizPrefix.render(el.prefixViz, step, {
      trace: LK_TRACES.prefix,
      state: state.prefix,
      setScenario(scenario) {
        state.prefix.scenario = scenario;
        state.prefix.snap = 0;
        stopPrefixPlay();
        renderPrefixViz();
        renderPrefixCounters();
      },
      setSnap(snap) {
        stopPrefixPlay();
        const data = LK_TRACES.prefix.scenarios[state.prefix.scenario];
        state.prefix.snap = Math.max(0, Math.min(data.snapshots.length - 1, snap));
        renderPrefixViz();
      },
      togglePlay() {
        if (state.prefix.playing) {
          stopPrefixPlay();
          renderPrefixViz();
          return;
        }
        const data = LK_TRACES.prefix.scenarios[state.prefix.scenario];
        if (state.prefix.snap >= data.snapshots.length - 1) state.prefix.snap = 0;
        state.prefix.playing = true;
        renderPrefixViz();
        state.prefix.timer = setInterval(() => {
          const d = LK_TRACES.prefix.scenarios[state.prefix.scenario];
          if (state.prefix.snap >= d.snapshots.length - 1) {
            stopPrefixPlay();
            renderPrefixViz();
            return;
          }
          state.prefix.snap += 1;
          renderPrefixViz();
        }, 900);
      },
      playing: state.prefix.playing,
    });
  }

  function stopPrefixPlay() {
    state.prefix.playing = false;
    if (state.prefix.timer) {
      clearInterval(state.prefix.timer);
      state.prefix.timer = null;
    }
  }

  function renderPrefixStep() {
    const steps = LK_DATA.prefix.steps;
    const step = steps[state.prefix.step];
    const n = steps.length;
    if (step.scenario && step.scenario !== state.prefix.scenario) {
      state.prefix.scenario = step.scenario;
      state.prefix.snap = 0;
    }
    const think = "";
    const docLink =
      '<a href="../' + LK_DATA.prefix.docFile + '" target="_blank" rel="noopener">' +
      LK_DATA.prefix.docFile + "</a>";
    const codeLink =
      '<a href="../' + LK_DATA.prefix.codeFile + '" target="_blank" rel="noopener">' +
      LK_DATA.prefix.codeFile + "</a>";
    el.prefixDoc.innerHTML =
      '<div class="doc-kicker">第 ' + (state.prefix.step + 1) + " / " + n + " 步 · " +
      esc(step.sectionTitle.replace(/`/g, "")) + "</div>" +
      "<h2>" + esc(step.title) + "</h2>" +
      '<p class="hint">' + esc(step.hint) + "</p>" + think +
      '<div class="doc-body">' + step.bodyHTML + "</div>" +
      '<div class="doc-foot">讲解：' + docLink + " · 代码：" + codeLink + "</div>";
    resetKvScroll(document.getElementById("prefix-right"));
    if (prefixCode) {
      const hasCode = applyStepCode(prefixCode, el.prefixCode, step);
      if (hasCode) prefixCode.scrollToLine(step.codeRanges[0].start);
      el.prefixCodeMeta.textContent = hasCode
        ? "本节代码 " + step.codeRanges.map((r) => r.start + "–" + r.end).join(" · ") + " 行"
        : "本节无代码";
    }
    renderPrefixViz();
    renderPrefixCounters();
    renderStepNext("prefix");
    updatePrefixDots();
    renderChapterNav("prefix");
    if (window.LK && LK.progress) {
      const l = LK.progress.lesson("prefix");
      if ((l.exploreMax || 0) < state.prefix.step + 1) {
        LK.progress.saveLesson("prefix", { exploreMax: state.prefix.step + 1 });
      }
      renderChapterNav("prefix");
    }
  }

  function setPrefixStep(i) {
    state.prefix.step = Math.max(0, Math.min(LK_DATA.prefix.steps.length - 1, i));
    const view = el.prefix;
    const main = view && view.querySelector(".kv-main");
    if (stageOf("prefix") === "explore" && main && main.classList.contains("flow-active")) {
      scrollKvTo("prefix", state.prefix.step);
    } else {
      renderPrefixStep();
      updateHash();
    }
  }

  function buildPrefixDots() {
    el.dots.innerHTML = "";
    LK_DATA.prefix.steps.forEach((step, i) => {
      const b = document.createElement("button");
      b.className = "dot";
      b.title = i + 1 + ". " + step.title;
      b.setAttribute("aria-label", b.title);
      b.addEventListener("click", () => setPrefixStep(i));
      el.dots.appendChild(b);
    });
  }

  function updatePrefixDots() {
    Array.from(el.dots.children).forEach((dot, i) => {
      dot.classList.toggle("active", i === state.prefix.step);
      dot.classList.toggle("done", i < state.prefix.step);
    });
  }

  /* -------------------------------------------------------- paged chapter */

  function renderPagedCounters() {
    const tr = LK_TRACES.paged;
    const snap = tr.snapshots[state.paged.snap] || tr.snapshots[0];
    const used = snap.blocks.filter((b) => b.owner).length;
    const waste = Object.keys(snap.lengths).reduce((sum, seq) => {
      const t = snap.tables[seq] || [];
      return sum + t.length * tr.block_size - snap.lengths[seq];
    }, 0);
    el.counters.innerHTML =
      "池占用 <b>" + used + "</b> / " + tr.num_blocks +
      '<span class="sep">|</span>空闲 <b>' + snap.free.length + "</b> 块" +
      '<span class="sep">|</span>内部浪费 <b>' + waste + "</b> slot" +
      '<span class="sep">|</span>' + esc(snap.label || "");
  }

  function renderPagedViz() {
    const step = LK_DATA.paged.steps[state.paged.step];
    LK.vizPaged.render(el.pagedViz, step, {
      trace: LK_TRACES.paged,
      state: state.paged,
      setSnap(snap) {
        stopPagedPlay();
        const tr = LK_TRACES.paged;
        state.paged.snap = Math.max(0, Math.min(tr.snapshots.length - 1, snap));
        renderPagedViz();
        renderPagedCounters();
      },
      setSel(seq) {
        state.paged.sel = seq;
        renderPagedViz();
      },
      setBlock(block) {
        const snap = LK_TRACES.paged.snapshots[state.paged.snap] || LK_TRACES.paged.snapshots[0];
        const hit = snap.blocks.find((b) => b.id === block);
        state.paged.block = state.paged.block === block ? null : block;
        if (state.paged.block !== null && hit && hit.owner) state.paged.sel = hit.owner;
        renderPagedViz();
      },
      togglePlay() {
        if (state.paged.playing) {
          stopPagedPlay();
          renderPagedViz();
          return;
        }
        const tr = LK_TRACES.paged;
        if (state.paged.snap >= tr.snapshots.length - 1) state.paged.snap = 0;
        state.paged.playing = true;
        renderPagedViz();
        state.paged.timer = setInterval(() => {
          const d = LK_TRACES.paged;
          if (state.paged.snap >= d.snapshots.length - 1) {
            stopPagedPlay();
            renderPagedViz();
            return;
          }
          state.paged.snap += 1;
          renderPagedViz();
          renderPagedCounters();
        }, 950);
      },
      playing: state.paged.playing,
    });
  }

  function stopPagedPlay() {
    state.paged.playing = false;
    if (state.paged.timer) {
      clearInterval(state.paged.timer);
      state.paged.timer = null;
    }
  }

  function renderPagedStep() {
    const steps = LK_DATA.paged.steps;
    const step = steps[state.paged.step];
    const n = steps.length;
    const docLink =
      '<a href="../' + LK_DATA.paged.docFile + '" target="_blank" rel="noopener">' +
      LK_DATA.paged.docFile + "</a>";
    const codeLink =
      '<a href="../' + LK_DATA.paged.codeFile + '" target="_blank" rel="noopener">' +
      LK_DATA.paged.codeFile + "</a>";
    el.pagedDoc.innerHTML =
      '<div class="doc-kicker">第 ' + (state.paged.step + 1) + " / " + n + " 步 · " +
      esc(step.sectionTitle.replace(/`/g, "")) + "</div>" +
      "<h2>" + esc(step.title) + "</h2>" +
      '<p class="hint">' + esc(step.hint) + "</p>" +
      '<div class="doc-body">' + step.bodyHTML + "</div>" +
      '<div class="doc-foot">讲解：' + docLink + " · 代码：" + codeLink + "</div>";
    resetKvScroll(document.getElementById("paged-right"));
    if (pagedCode) {
      const hasCode = applyStepCode(pagedCode, el.pagedCode, step);
      if (hasCode) pagedCode.scrollToLine(step.codeRanges[0].start);
      el.pagedCodeMeta.textContent = hasCode
        ? "本节代码 " + step.codeRanges.map((r) => r.start + "–" + r.end).join(" · ") + " 行"
        : "本节无代码";
    }
    renderPagedViz();
    renderPagedCounters();
    renderStepNext("paged");
    updatePagedDots();
    renderChapterNav("paged");
    if (window.LK && LK.progress) {
      const l = LK.progress.lesson("paged");
      if ((l.exploreMax || 0) < state.paged.step + 1) {
        LK.progress.saveLesson("paged", { exploreMax: state.paged.step + 1 });
      }
      renderChapterNav("paged");
    }
  }

  function setPagedStep(i) {
    state.paged.step = Math.max(0, Math.min(LK_DATA.paged.steps.length - 1, i));
    const view = el.paged;
    const main = view && view.querySelector(".kv-main");
    if (stageOf("paged") === "explore" && main && main.classList.contains("flow-active")) {
      scrollKvTo("paged", state.paged.step);
    } else {
      renderPagedStep();
      updateHash();
    }
  }

  function buildPagedDots() {
    el.dots.innerHTML = "";
    LK_DATA.paged.steps.forEach((step, i) => {
      const b = document.createElement("button");
      b.className = "dot";
      b.title = i + 1 + ". " + step.title;
      b.setAttribute("aria-label", b.title);
      b.addEventListener("click", () => setPagedStep(i));
      el.dots.appendChild(b);
    });
  }

  function updatePagedDots() {
    Array.from(el.dots.children).forEach((dot, i) => {
      dot.classList.toggle("active", i === state.paged.step);
      dot.classList.toggle("done", i < state.paged.step);
    });
  }

  /* ---------------------------------------------------------- step nav */

  function navStore() {
    try {
      return JSON.parse(localStorage.getItem("whykvc.stepnav") || "{}") || {};
    } catch (e) {
      return {};
    }
  }

  function saveNavStore() {
    try {
      localStorage.setItem("whykvc.stepnav", JSON.stringify(state.navCollapsed));
    } catch (e) {
      /* localStorage unavailable */
    }
  }

  function initNav() {
    const saved = navStore();
    ALL_CHAPTERS.forEach((ch) => {
      state.navCollapsed[ch] =
        typeof saved[ch] === "boolean" ? saved[ch] : window.innerWidth <= 980;
    });
  }

  function applyNavCollapsed(ch) {
    const view = el[ch];
    if (!view) return;
    const chapter = view.querySelector(".chapter");
    if (chapter) chapter.classList.toggle("nav-collapsed", !!state.navCollapsed[ch]);
    const nav = el.stepNavs[ch];
    const btn = nav && nav.querySelector(".stepnav-toggle");
    if (btn) {
      btn.textContent = state.navCollapsed[ch] ? "›" : "‹";
      btn.setAttribute("aria-expanded", String(!state.navCollapsed[ch]));
      btn.setAttribute("aria-label", state.navCollapsed[ch] ? "展开章节导航" : "收起章节导航");
    }
  }

  function toggleNav(ch) {
    state.navCollapsed[ch] = !state.navCollapsed[ch];
    applyNavCollapsed(ch);
    saveNavStore();
  }

  function renderChapterNav(ch) {
    const nav = el.stepNavs[ch];
    if (!nav) return;
    const steps = chapterSteps(ch);
    const idx = chapterStepIndex(ch);
    const stage = stageOf(ch);
    const lesson = LK.progress.lesson(ch);
    const stageIdx = STAGE_NAMES.indexOf(stage);

    const attrCards = (D.lessons[ch] && D.lessons[ch].attribution) || [];
    const attrDone = !attrCards.length || !!lesson.attributionSubmitted;
    const stageDefs = [
      ["think", "请回答", lesson.think ? "✓" : ""],
      ["explore", "探索阅读", (lesson.exploreMax || 0) + " / " + steps.length],
      ["reflect", "反思与总结", lesson.summary ? "✓" : attrDone ? "待总结" : "待归因"],
    ];
    const stagesHTML = stageDefs
      .map(
        ([id, label, sub], i) =>
          '<button class="stage-nav-item' + (id === stage ? " on" : "") +
          (sub === "✓" ? " done" : "") + '" data-nav-stage="' + id + '" title="' + label +
          (sub ? "：" + sub : "") + '">' +
          '<span class="sid">0' + (i + 1) + "</span>" +
          '<span class="stt">' + label + "</span>" +
          '<em>' + sub + "</em></button>"
      )
      .join("");
    const showSteps = stage === "explore";
    const isDoc = DOC_CHAPTERS.includes(ch);
    const headCount = showSteps
      ? idx + 1 + " / " + steps.length
      : stageIdx + 1 + " / " + STAGE_NAMES.length;
    nav.innerHTML =
      '<div class="stepnav-head"><span>章节导航</span><b>' + headCount + "</b>" +
      '<button class="stepnav-toggle" type="button" aria-expanded="true" aria-label="收起章节导航">‹</button></div>' +
      '<div class="stage-nav">' + stagesHTML + "</div>" +
      (showSteps
        ? '<div class="stepnav-section">探索步骤</div>' +
          '<ol class="stepnav-list">' +
          steps
            .map(
              (st, i) =>
                '<li><button class="stepnav-item' + (i === idx ? " active" : "") +
                (i < idx ? " done" : "") + '" data-nav-i="' + i + '" title="' + i + 1 + ". " +
                String(st.title).replace(/"/g, "&quot;") + '">' +
                '<span class="no">' + (i < idx ? "✓" : i + 1) + "</span>" +
                '<span class="tt">' + esc(st.title) + "</span></button></li>"
            )
            .join("") +
          "</ol>"
        : '<p class="stepnav-hint">切换到「探索阅读」查看 ' + steps.length + " 个步骤。</p>");

    nav.querySelector(".stepnav-toggle").addEventListener("click", () => toggleNav(ch));
    nav.querySelectorAll("[data-nav-stage]").forEach((b) =>
      b.addEventListener("click", () => setStage(ch, b.dataset.navStage))
    );
    nav.querySelectorAll("[data-nav-i]").forEach((b) =>
      b.addEventListener("click", () => {
        const i = +b.dataset.navI;
        if (isDoc) scrollDocTo(ch, i);
        else if (ch === "kv") setStep(i);
        else if (ch === "prefix") setPrefixStep(i);
        else setPagedStep(i);
      })
    );
    applyNavCollapsed(ch);
    const list = nav.querySelector(".stepnav-list");
    const active = list && list.querySelector(".active");
    if (list && active) {
      const offset =
        active.getBoundingClientRect().top - list.getBoundingClientRect().top + list.scrollTop;
      list.scrollTop = Math.max(0, offset - list.clientHeight / 2 + active.offsetHeight / 2);
    }
  }

  /* ------------------------------------------------------- lesson stages */

  function docSectionSteps(ch) {
    const d = D[ch];
    const meta = d.sections || {};
    return DOC_SECTIONS[ch].map((key) => {
      const raw = (meta[key] && meta[key].title) || (d[key] && d[key].title) || key;
      return { title: String(raw).replace(/^\d+\s*[·.]?\s*/, "") };
    });
  }

  function chapterSteps(ch) {
    if (DOC_CHAPTERS.includes(ch)) return docSectionSteps(ch);
    return ch === "kv" ? kv.steps : ch === "prefix" ? LK_DATA.prefix.steps : LK_DATA.paged.steps;
  }

  function chapterStepIndex(ch) {
    if (DOC_CHAPTERS.includes(ch)) return state.docStep[ch] || 0;
    return ch === "kv" ? state.step : ch === "prefix" ? state.prefix.step : state.paged.step;
  }

  function setChapterStep(ch, i) {
    const max = chapterSteps(ch).length - 1;
    const v = Math.max(0, Math.min(max, i));
    if (DOC_CHAPTERS.includes(ch)) state.docStep[ch] = v;
    else if (ch === "kv") state.step = v;
    else if (ch === "prefix") state.prefix.step = v;
    else state.paged.step = v;
  }

  function stageOf(ch) {
    return state.stages[ch] || "think";
  }

  function initStages() {
    ALL_CHAPTERS.forEach((ch) => {
      const l = LK.progress.lesson(ch);
      state.stages[ch] = l.summary || l.reflect ? "reflect" : l.think ? "explore" : "think";
    });
  }

  function setStage(ch, stage) {
    if (stage === "summary") stage = "reflect";
    if (!STAGE_NAMES.includes(stage)) return;
    state.stages[ch] = stage;
    if (stage !== "explore") {
      if (ch === "prefix") stopPrefixPlay();
      if (ch === "paged") stopPagedPlay();
    }
    if (DOC_CHAPTERS.includes(ch)) {
      renderDocStage(ch);
      updateHash();
      return;
    }
    renderChapterStage(ch);
    updateHash();
  }

  /* ------------------------------------------------- KV continuous flow */

  const kvFlowStops = {};

  function kvFlowCfg(ch) {
    const d = ch === "kv" ? kv : ch === "prefix" ? LK_DATA.prefix : LK_DATA.paged;
    return { d, steps: d.steps, n: d.steps.length, codeFile: d.codeFile, docFile: d.docFile };
  }

  function setFlowStepState(ch, i) {
    if (ch === "kv") state.step = i;
    else if (ch === "prefix") state.prefix.step = i;
    else state.paged.step = i;
  }

  function pad2(n) {
    return String(n).padStart(2, "0");
  }

  function kvFlowSectionHTML(ch, step, i, cfg) {
    const ranges = step.codeRanges || [];
    const rangeLabel = ranges.length
      ? "本节代码 " + ranges.slice().sort((a, b) => a.start - b.start)
        .map((r) => r.start + "–" + r.end).join(" · ") + " 行"
      : "";
    const code = ranges.length
      ? '<div class="kv-sec-code"><div class="pane-head"><span class="file">' + esc(cfg.codeFile) +
        '</span><span class="pane-meta">' + rangeLabel + '</span></div>' +
        '<div class="code-scroll" data-kv-code="' + i + '"></div></div>'
      : "";
    const think = step.think
      ? '<details class="thinkbox step-think"><summary>请回答：' + esc(step.think.q) +
        "</summary><p>" + esc(step.think.a) + "</p></details>"
      : "";
    const viz = step.viz && step.viz !== "none"
      ? '<div class="kv-sec-viz" data-kv-viz="' + i + '"></div>'
      : "";
    return '<section class="kv-sec" data-kv-sec="' + i + '" id="' + ch + '-sec-' + i + '">' +
      '<div class="sec-pos">' + pad2(i + 1) + " / " + pad2(cfg.n) + "</div>" +
      "<h2>" + esc(step.title) + "</h2>" +
      '<p class="hint">' + esc(step.hint) + "</p>" +
      think +
      code +
      '<div class="doc-body">' + step.bodyHTML + "</div>" +
      '<div class="doc-foot">讲解：<a href="../' + esc(cfg.docFile) + '" target="_blank" rel="noopener">' +
      esc(cfg.docFile) + '</a> · 代码：<a href="../' + esc(cfg.codeFile) +
      '" target="_blank" rel="noopener">' + esc(cfg.codeFile) + "</a></div>" +
      viz +
      "</section>";
  }

  function stopKvFlowViz(ch) {
    const arr = kvFlowStops[ch];
    if (!arr) return;
    arr.forEach((fn) => fn());
    kvFlowStops[ch] = [];
  }

  function registerKvVizStop(ch, fn) {
    if (!kvFlowStops[ch]) kvFlowStops[ch] = [];
    kvFlowStops[ch].push(fn);
  }

  function mountKvFlowViz(ch, i, slot, step) {
    if (!slot || !step.viz || step.viz === "none") return;

    if (ch === "kv") {
      let genStep = MAX_GEN;
      let playing = false;
      let timer = null;
      let ctx = null;
      const rerender = () => LK.viz.render(slot, step, ctx);
      const stop = () => {
        if (timer) { clearInterval(timer); timer = null; }
        if (playing) { playing = false; if (slot.isConnected) rerender(); }
      };
      ctx = {
        trace: TR,
        get genStep() { return genStep; },
        get playing() { return playing; },
        setGenStep(v) {
          genStep = Math.max(0, Math.min(MAX_GEN, v));
          rerender();
        },
        togglePlay() {
          if (playing) { stop(); return; }
          if (genStep >= MAX_GEN) genStep = 0;
          playing = true;
          rerender();
          timer = setInterval(() => {
            if (genStep >= MAX_GEN) { stop(); return; }
            genStep += 1;
            rerender();
          }, 420);
        },
      };
      registerKvVizStop(ch, stop);
      rerender();
      return;
    }

    if (ch === "prefix") {
      if (step.viz === "reuse") {
        LK.workshop.renderReuse(slot, LK_TRACES.kv.reuse, { compact: true });
        return;
      }
      const local = {
        scenario: step.scenario || "shared",
        snap: 0,
      };
      if (!LK_TRACES.prefix.scenarios[local.scenario]) local.scenario = "shared";
      let playing = false;
      let timer = null;
      let ctx = null;
      const data = () => LK_TRACES.prefix.scenarios[local.scenario];
      const rerender = () => LK.vizPrefix.render(slot, step, ctx);
      const stop = () => {
        if (timer) { clearInterval(timer); timer = null; }
        if (playing) { playing = false; if (slot.isConnected) rerender(); }
      };
      ctx = {
        trace: LK_TRACES.prefix,
        state: local,
        get playing() { return playing; },
        setScenario(sc) {
          local.scenario = LK_TRACES.prefix.scenarios[sc] ? sc : "shared";
          local.snap = 0;
          stop();
          rerender();
        },
        setSnap(snap) {
          local.snap = Math.max(0, Math.min(data().snapshots.length - 1, snap));
          stop();
          rerender();
        },
        togglePlay() {
          if (playing) { stop(); return; }
          if (local.snap >= data().snapshots.length - 1) local.snap = 0;
          playing = true;
          rerender();
          timer = setInterval(() => {
            if (local.snap >= data().snapshots.length - 1) { stop(); return; }
            local.snap += 1;
            rerender();
          }, 420);
        },
      };
      registerKvVizStop(ch, stop);
      rerender();
      return;
    }

    const local = { snap: 0, sel: "seq-b", block: null };
    let playing = false;
    let timer = null;
    let ctx = null;
    const snaps = () => LK_TRACES.paged.snapshots;
    const rerender = () => LK.vizPaged.render(slot, step, ctx);
    const stop = () => {
      if (timer) { clearInterval(timer); timer = null; }
      if (playing) { playing = false; if (slot.isConnected) rerender(); }
    };
    ctx = {
      trace: LK_TRACES.paged,
      state: local,
      get playing() { return playing; },
      setSnap(snap) {
        local.snap = Math.max(0, Math.min(snaps().length - 1, snap));
        stop();
        rerender();
      },
      setSel(sel) {
        local.sel = sel;
        stop();
        rerender();
      },
      setBlock(block) {
        local.block = block;
        stop();
        rerender();
      },
      togglePlay() {
        if (playing) { stop(); return; }
        local.snap = 0;
        playing = true;
        rerender();
        timer = setInterval(() => {
          if (local.snap >= snaps().length - 1) { stop(); return; }
          local.snap += 1;
          rerender();
        }, 420);
      },
    };
    registerKvVizStop(ch, stop);
    rerender();
  }

  function buildKvFlow(ch) {
    const view = el[ch];
    const main = view && view.querySelector(".kv-main");
    const flow = main && main.querySelector(".kv-flow");
    if (!flow) return;
    stopKvFlowViz(ch);
    const cfg = kvFlowCfg(ch);
    const d = cfg.d;
    flow.innerHTML = '<div class="kv-page">' +
      '<div class="read-pos" data-flow-pos>' + docPosText(ch, chapterStepIndex(ch), cfg.n) + "</div>" +
      '<div class="doc-kicker">学习章节 · ' + esc(d.title) + "</div>" +
      '<h1 class="page-title">' + esc(d.title) + "</h1>" +
      '<p class="page-lead">连续长页：向下滚动查看全部小节；左侧步骤导航定位，顶部显示当前进度。</p>' +
      cfg.steps.map((step, i) => kvFlowSectionHTML(ch, step, i, cfg)).join("") +
      "</div>";
    flow.querySelectorAll("[data-kv-code]").forEach((slot) => {
      const i = +slot.dataset.kvCode;
      const step = cfg.steps[i];
      if (step && step.codeRanges && step.codeRanges.length) {
        LK.codeview.mountRanges(slot, cfg.d.lines, step.codeRanges);
      }
    });
    cfg.steps.forEach((step, i) => {
      const slot = flow.querySelector('[data-kv-viz="' + i + '"]');
      if (slot) mountKvFlowViz(ch, i, slot, step);
    });
    flow.dataset.built = "1";
    initKvFlowScroll(ch);
  }

  function initKvFlowScroll(ch) {
    const view = el[ch];
    const main = view && view.querySelector(".kv-main");
    if (!main || main.dataset.flowBound === "1") return;
    main.dataset.flowBound = "1";
    main.addEventListener("scroll", () => {
      if (state.view !== ch || stageOf(ch) !== "explore" || !main.classList.contains("flow-active")) return;
      const flow = main.querySelector(".kv-flow");
      if (!flow) return;
      const secs = Array.from(flow.querySelectorAll(".kv-sec"));
      if (!secs.length) return;
      const top = main.getBoundingClientRect().top + 96;
      let idx = 0;
      secs.forEach((sec, i) => {
        if (sec.getBoundingClientRect().top <= top) idx = i;
      });
      if (main.scrollTop + main.clientHeight >= main.scrollHeight - 4) idx = secs.length - 1;
      if (idx === chapterStepIndex(ch)) return;
      setFlowStepState(ch, idx);
      const pos = flow.querySelector("[data-flow-pos]");
      if (pos) pos.textContent = docPosText(ch, idx, secs.length);
      if (LK.progress) {
        LK.progress.recordStep(ch + ":" + (idx + 1));
        const lesson = LK.progress.lesson(ch);
        if ((lesson.exploreMax || 0) < idx + 1) LK.progress.saveLesson(ch, { exploreMax: idx + 1 });
      }
      renderChapterNav(ch);
      updateHash();
    }, { passive: true });
  }

  function scrollKvTo(ch, i) {
    const cfg = kvFlowCfg(ch);
    const idx = Math.max(0, Math.min(cfg.n - 1, i));
    setFlowStepState(ch, idx);
    const view = el[ch];
    const main = view && view.querySelector(".kv-main");
    const flow = main && main.querySelector(".kv-flow");
    const sec = flow && flow.querySelector('[data-kv-sec="' + idx + '"]');
    if (main && idx === 0) {
      main.scrollTo({ top: 0, behavior: "auto" });
    } else if (main && sec) {
      const posEl = flow.querySelector("[data-flow-pos]");
      const offset = (posEl ? posEl.offsetHeight : 0) + 10;
      const top = sec.getBoundingClientRect().top - main.getBoundingClientRect().top + main.scrollTop;
      main.scrollTo({ top: Math.max(0, top - offset), behavior: "auto" });
    }
    const pos = flow && flow.querySelector("[data-flow-pos]");
    if (pos) pos.textContent = docPosText(ch, idx, cfg.n);
    if (LK.progress) {
      LK.progress.recordStep(ch + ":" + (idx + 1));
      const lesson = LK.progress.lesson(ch);
      if ((lesson.exploreMax || 0) < idx + 1) LK.progress.saveLesson(ch, { exploreMax: idx + 1 });
    }
    renderChapterNav(ch);
    updateHash();
  }

  function renderKvFlowStage(ch) {
    const view = el[ch];
    const main = view && view.querySelector(".kv-main");
    const flow = main && main.querySelector(".kv-flow");
    if (!main || !flow) return;
    main.classList.add("flow-active");
    const timeline = document.querySelector(".timeline");
    if (timeline) timeline.style.display = "none";
    if (!flow.dataset.built) buildKvFlow(ch);
    renderChapterNav(ch);
    scrollKvTo(ch, chapterStepIndex(ch));
  }

  function renderChapterStage(ch) {
    if (DOC_CHAPTERS.includes(ch)) {
      renderDocStage(ch);
      return;
    }
    const view = el[ch];
    if (!view) return;
    STAGE_NAMES.forEach((s) => view.classList.toggle("stage-" + s, stageOf(ch) === s));
    renderChapterNav(ch);
    const timeline = document.querySelector(".timeline");
    const docEl = ch === "kv" ? el.doc : ch === "prefix" ? el.prefixDoc : el.pagedDoc;
    const vizEl = ch === "kv" ? el.viz : ch === "prefix" ? el.prefixViz : el.pagedViz;
    const kvMain = view.querySelector(".kv-main");
    if (kvMain) kvMain.scrollTop = 0;
    if (stageOf(ch) === "explore") {
      renderKvFlowStage(ch);
      return;
    }
    if (kvMain) kvMain.classList.remove("flow-active");
    stopKvFlowViz(ch);
    const nextHost = stepNextHost(ch);
    if (nextHost) nextHost.innerHTML = "";
    if (timeline) timeline.style.display = "none";
    if (vizEl) vizEl.innerHTML = "";
    const lesson = LK_DATA.lessons[ch];
    if (stageOf(ch) === "think") {
      LK.workshop.renderStageThink(docEl, ch, lesson.think, () => renderChapterNav(ch), () => {
        setChapterStep(ch, 0);
        setStage(ch, "explore");
      });
    } else {
      LK.workshop.renderStageReview(
        docEl, ch, lesson.reflect, lesson.attribution, lesson.summary,
        () => renderChapterNav(ch)
      );
    }
  }


  /* ------------------------------------------------------- doc chapters */

  function docBody(ch) {
    return el[ch + "Body"];
  }

  function docMainEl(ch) {
    return el[ch + "Main"];
  }

  function docPosText(ch, idx, total) {
    return "探索阅读 · " + String(idx + 1).padStart(2, "0") + " / " + String(total).padStart(2, "0");
  }

  function injectDocSectionNav(ch, body) {
    body.querySelectorAll(".doc-sec-next, .sec-pos, .read-pos").forEach((n) => n.remove());
    const secs = Array.from(body.querySelectorAll("h2.sec-title"));
    if (!secs.length) return;
    const pos = document.createElement("div");
    pos.className = "read-pos";
    pos.dataset.docPos = "1";
    pos.textContent = docPosText(ch, chapterStepIndex(ch), secs.length);
    body.insertBefore(pos, body.firstChild);
    secs.forEach((h2, i) => {
      const first = h2.firstChild;
      if (first && first.nodeType === 3) {
        first.nodeValue = first.nodeValue.replace(/^\s*\d+\s*[·.]\s*/, "");
      }
      const badge = document.createElement("div");
      badge.className = "sec-pos";
      badge.textContent = String(i + 1).padStart(2, "0") + " / " + String(secs.length).padStart(2, "0");
      h2.parentNode.insertBefore(badge, h2);
    });
  }

  function renderDocStage(ch) {
    const view = el[ch];
    if (!view) return;
    stopDocChapter(ch);
    STAGE_NAMES.forEach((s) => view.classList.toggle("stage-" + s, stageOf(ch) === s));
    renderChapterNav(ch);
    const body = docBody(ch);
    const main = docMainEl(ch);
    if (!body || !main) return;
    const stage = stageOf(ch);

    if (stage === "explore") {
      renderDocChapter(ch, body);
      injectDocSectionNav(ch, body);
      bindDocScroll(ch);
      main.scrollTop = 0;
      const i = chapterStepIndex(ch);
      const secs = body.querySelectorAll("h2.sec-title");
      if (i > 0 && secs[i]) scrollMainToSection(ch, secs[i]);
      updateDocActive(ch);
      return;
    }

    const lesson = D.lessons[ch];
    if (stage === "think") {
      LK.workshop.renderStageThink(body, ch, lesson.think, () => renderChapterNav(ch), () => {
        setChapterStep(ch, 0);
        setStage(ch, "explore");
      });
    } else {
      LK.workshop.renderStageReview(
        body, ch, lesson.reflect, lesson.attribution, lesson.summary,
        () => renderChapterNav(ch)
      );
    }
    main.scrollTop = 0;
  }

  function docModule(ch) {
    const get = DOC_MODULES[ch];
    return get ? get() : null;
  }

  function stopDocChapter(ch) {
    const mod = docModule(ch);
    if (mod && typeof mod.stop === "function") mod.stop();
  }

  function renderDocChapter(ch, body) {
    const mod = docModule(ch);
    if (!mod || typeof mod.render !== "function") {
      body.innerHTML = '<p class="note">章节渲染器未加载：检查 assets/' + DOC_RENDERERS[ch] + '.js。</p>';
      return;
    }
    mod.render(body, D[ch], world);
  }

  function bindDocScroll(ch) {
    const main = docMainEl(ch);
    if (!main || main.dataset.docBound === "1") return;
    main.dataset.docBound = "1";
    main.addEventListener("scroll", () => {
      if (state.view !== ch || stageOf(ch) !== "explore") return;
      updateDocActive(ch);
    }, { passive: true });
  }

  function currentDocIndex(ch) {
    const main = docMainEl(ch);
    const body = docBody(ch);
    if (!main || !body) return 0;
    const secs = Array.from(body.querySelectorAll("h2.sec-title"));
    if (!secs.length) return 0;
    const top = main.getBoundingClientRect().top + 96;
    let idx = 0;
    secs.forEach((sec, i) => {
      if (sec.getBoundingClientRect().top <= top) idx = i;
    });
    if (main.scrollTop + main.clientHeight >= main.scrollHeight - 4) idx = secs.length - 1;
    return idx;
  }

  function updateDocActive(ch) {
    const nav = el.stepNavs[ch];
    const steps = chapterSteps(ch);
    if (!nav || !steps.length) return;
    if (Date.now() < (state.docLock[ch] || 0)) return;
    const idx = Math.max(0, Math.min(steps.length - 1, currentDocIndex(ch)));
    const prev = state.docStep[ch];
    state.docStep[ch] = idx;
    if (LK.progress) LK.progress.recordStep(ch + ":" + (idx + 1));
    if (prev !== idx) updateHash();
    nav.querySelectorAll(".stepnav-item").forEach((b, i) => {
      b.classList.toggle("active", i === idx);
      b.classList.toggle("done", i < idx);
    });
    const head = nav.querySelector(".stepnav-head > b");
    if (head) head.textContent = idx + 1 + " / " + steps.length;
    const em = nav.querySelector('[data-nav-stage="explore"] em');
    const pos = docBody(ch) && docBody(ch).querySelector("[data-doc-pos]");
    if (pos) pos.textContent = docPosText(ch, idx, steps.length);
    const lesson = LK.progress.lesson(ch);
    const max = Math.max(lesson.exploreMax || 0, idx + 1);
    if (max !== (lesson.exploreMax || 0)) LK.progress.saveLesson(ch, { exploreMax: max });
    if (em) em.textContent = max + " / " + steps.length;
    syncNavGroups(ch);
  }

  function scrollMainToSection(ch, sec) {
    const main = docMainEl(ch);
    if (!main || !sec) return;
    const posEl = main.querySelector(".read-pos");
    const offset = (posEl ? posEl.offsetHeight : 0) + 10;
    const top = main.scrollTop + sec.getBoundingClientRect().top - main.getBoundingClientRect().top - offset;
    main.scrollTo({ top: Math.max(0, top), behavior: "auto" });
  }

  function scrollDocTo(ch, i) {
    const body = docBody(ch);
    const main = docMainEl(ch);
    if (!body || !main) return;
    const secs = Array.from(body.querySelectorAll("h2.sec-title"));
    if (!secs.length) return;
    const v = Math.max(0, Math.min(secs.length - 1, i));
    state.docStep[ch] = v;
    state.docLock[ch] = Date.now() + 300;
    if (v === 0) main.scrollTo({ top: 0, behavior: "auto" });
    else scrollMainToSection(ch, secs[v]);
    const lesson = LK.progress.lesson(ch);
    const max = Math.max(lesson.exploreMax || 0, v + 1);
    if (max !== (lesson.exploreMax || 0)) LK.progress.saveLesson(ch, { exploreMax: max });
    if (LK.progress) LK.progress.recordStep(ch + ":" + (v + 1));
    const nav = el.stepNavs[ch];
    if (nav) {
      nav.querySelectorAll(".stepnav-item").forEach((b, k) => {
        b.classList.toggle("active", k === v);
        b.classList.toggle("done", k < v);
      });
      const head = nav.querySelector(".stepnav-head > b");
      if (head) head.textContent = v + 1 + " / " + secs.length;
      const em = nav.querySelector('[data-nav-stage="explore"] em');
      if (em) em.textContent = max + " / " + secs.length;
    }
    updateHash();
  }

  /* ---------------------------------------------------------- dots/hash */

  function goPrevChapter(ch) {
    const i = LEARNING_ORDER.indexOf(ch);
    if (i <= 0) return;
    const prev = LEARNING_ORDER[i - 1];
    state.stages[prev] = "explore";
    setChapterStep(prev, chapterSteps(prev).length - 1);
    setView(prev);
    updateHash();
  }

  function navStep(delta) {
    const view = state.view;
    if (!ALL_CHAPTERS.includes(view)) return;
    const stage = stageOf(view);
    const steps = chapterSteps(view);
    const idx = chapterStepIndex(view);

    if (delta > 0) {
      if (stage === "think") {
        setChapterStep(view, 0);
        setStage(view, "explore");
      } else if (stage === "reflect") {
        goNextChapter(view);
      } else if (idx < steps.length - 1) {
        if (DOC_CHAPTERS.includes(view)) scrollDocTo(view, idx + 1);
        else if (view === "kv") setStep(idx + 1);
        else if (view === "prefix") setPrefixStep(idx + 1);
        else setPagedStep(idx + 1);
      } else {
        goNextChapter(view);
      }
      return;
    }

    if (stage === "think") {
      goPrevChapter(view);
    } else if (stage === "reflect") {
      state.stages[view] = "explore";
      setChapterStep(view, steps.length - 1);
      setStage(view, "explore");
    } else if (idx > 0) {
      if (DOC_CHAPTERS.includes(view)) scrollDocTo(view, idx - 1);
      else if (view === "kv") setStep(idx - 1);
      else if (view === "prefix") setPrefixStep(idx - 1);
      else setPagedStep(idx - 1);
    } else {
      setStage(view, "think");
    }
  }

  function buildDots() {
    el.dots.innerHTML = "";
    kv.steps.forEach((step, i) => {
      const b = document.createElement("button");
      b.className = "dot";
      b.title = i + 1 + ". " + step.title;
      b.setAttribute("aria-label", b.title);
      b.addEventListener("click", () => setStep(i));
      el.dots.appendChild(b);
    });
  }

  function updateDots() {
    Array.from(el.dots.children).forEach((dot, i) => {
      dot.classList.toggle("active", i === state.step);
      dot.classList.toggle("done", i < state.step);
    });
  }

  function updateHash() {
    const view = VIEWS.includes(state.view) ? state.view : "think";
    let hash = "#" + view;
    if (CHAPTERS.includes(view) || DOC_CHAPTERS.includes(view)) {
      const ch = view;
      const stage = stageOf(ch);
      hash = stage === "explore" ? "#" + ch + "/" + (chapterStepIndex(ch) + 1) : "#" + ch + "/" + stage;
    }
    if (location.hash !== hash) history.replaceState(null, "", hash);
  }

  /* ------------------------------------------------------ topbar dropdowns */

  function closeNavMenus(except) {
    navGroups.forEach((g) => {
      if (g === except || g.menu.hidden) return;
      g.menu.hidden = true;
      g.trigger.setAttribute("aria-expanded", "false");
    });
  }

  function resetMenuPos(g) {
    g.menu.style.position = "";
    g.menu.style.left = "";
    g.menu.style.right = "";
    g.menu.style.top = "";
    g.menu.style.width = "";
  }

  function placeNavMenu(g) {
    resetMenuPos(g);
    if (window.innerWidth <= 980) {
      // Fixed + full width on mobile: an absolutely positioned menu that
      // starts off-screen widens the layout viewport before it is clamped.
      const r = g.trigger.getBoundingClientRect();
      g.menu.style.position = "fixed";
      g.menu.style.left = "12px";
      g.menu.style.right = "12px";
      g.menu.style.top = Math.round(r.bottom + 6) + "px";
      g.menu.style.width = "auto";
      return;
    }
    const rect = g.menu.getBoundingClientRect();
    const overflow = rect.right - (window.innerWidth - 12);
    if (overflow > 0) g.menu.style.left = -Math.min(overflow, rect.left - 12) + "px";
  }

  function openNavMenu(g) {
    closeNavMenus(g);
    g.menu.hidden = false;
    g.trigger.setAttribute("aria-expanded", "true");
    placeNavMenu(g);
    const first = g.menu.querySelector('[role="menuitem"]');
    if (first) first.focus();
  }

  function toggleNavMenu(g) {
    if (g.menu.hidden) {
      openNavMenu(g);
    } else {
      g.menu.hidden = true;
      g.trigger.setAttribute("aria-expanded", "false");
    }
  }

  function syncNavGroups(view) {
    let stepNo = null;
    if (CHAPTERS.includes(view) || DOC_CHAPTERS.includes(view)) {
      stepNo = chapterStepIndex(view) + 1;
    }
    navGroups.forEach((g) => {
      const items = Array.from(g.menu.querySelectorAll("[data-view]"));
      const sameView = items.filter((b) => b.dataset.view === view);
      const stepped = sameView.filter((b) => b.dataset.step);
      const plain = sameView.filter((b) => !b.dataset.step);
      const active =
        stepped.filter((b) => +b.dataset.step === stepNo)[0] ||
        (stepped.length ? null : plain[0] || sameView[0] || null);
      items.forEach((b) => b.setAttribute("aria-current", b === active ? "true" : "false"));
      g.trigger.classList.toggle("active", !!active);
      g.label.textContent = active ? g.base + " · " + active.dataset.short : g.base;
    });
  }

  /* -------------------------------------------------------------- views */

  function setView(view) {
    if (!VIEWS.includes(view)) view = "think";
    DOC_CHAPTERS.forEach((ch) => stopDocChapter(ch));
    if (state.playing) stopPlay(false);
    if (CHAPTERS.includes(state.view) && state.view !== view) stopKvFlowViz(state.view);
    state.view = view;
    document.body.dataset.view = view;
    const timeline = document.querySelector(".timeline");
    if (timeline) timeline.style.display = "";
    if (LK.progress) LK.progress.recordView(view);
    closeNavMenus();
    el.tabs.forEach((t) => {
      const on = t.dataset.view === view;
      t.classList.toggle("active", on);
      t.setAttribute("aria-current", on ? "true" : "false");
    });
    syncNavGroups(view);
    VIEWS.forEach((v) => el[v] && el[v].classList.toggle("active", v === view));
    if (view === "kv") {
      buildDots();
      renderChapterStage("kv");
    } else if (view === "prefix") {
      buildPrefixDots();
      renderChapterStage("prefix");
    } else if (view === "paged") {
      buildPagedDots();
      renderChapterStage("paged");
    } else if (view === "think") {
      LK.workshop.renderThink(el.quiz, {
        quiz: D.think.quiz,
        context: D.anatomy.context,
        presets: D.anatomy.presets,
      });
    } else if (view === "calc") {
      LK.calc.render(el.calcBody, {
        presets: D.anatomy.presets,
        context: D.anatomy.context,
      });
    } else if (DOC_CHAPTERS.includes(view)) {
      renderDocStage(view);
    } else if (view === "feedback") {
      LK.workshop.renderFeedback(el.feedbackBody, {
        preQuiz: D.think.quiz,
        postTest: D.think.postTest,
        takeaway: D.reflect.takeaway,
        summary: D.reflect.feedbackSummary,
        totalSteps:
          kv.steps.length + D.prefix.steps.length + D.paged.steps.length +
          DOC_CHAPTERS.reduce((n, ch) => n + docSectionSteps(ch).length, 0),
      });
    }
  }

  function route() {
    const hash = location.hash.replace(/^#/, "");
    for (const ch of ALL_CHAPTERS) {
      if (hash === ch || hash.startsWith(ch + "/")) {
        const rest = hash.slice(ch.length + 1);
        const stage = rest === "summary" ? "reflect" : rest;
        if (STAGE_NAMES.includes(stage)) {
          state.stages[ch] = stage;
        } else if (/^\d+$/.test(rest)) {
          state.stages[ch] = "explore";
          setChapterStep(ch, +rest - 1);
        }
        setView(ch);
        updateHash();
        return;
      }
    }
    if (hash === "reuse") {
      state.stages.prefix = "explore";
      setChapterStep("prefix", 10);
      setView("prefix");
      updateHash();
      return;
    }
    if (hash === "reflect") {
      state.stages.kv = "reflect";
      setView("kv");
      updateHash();
      return;
    }
    const view = VIEWS.includes(hash) ? hash : "think";
    setView(view);
    updateHash();
  }

  /* -------------------------------------------------------------- events */

  function bind() {
    el.tabs.forEach((tab) => {
      if (tab.disabled) return;
      tab.addEventListener("click", () => {
        if (!VIEWS.includes(tab.dataset.view)) return;
        if (tab.dataset.step) {
          const target = tab.dataset.view;
          if (CHAPTERS.includes(target) || DOC_CHAPTERS.includes(target)) {
            state.stages[target] = "explore";
            setChapterStep(target, +tab.dataset.step - 1);
          }
        }
        setView(tab.dataset.view);
        updateHash();
      });
    });

    navGroups.forEach((g) => {
      g.trigger.addEventListener("click", () => toggleNavMenu(g));
      g.trigger.addEventListener("keydown", (e) => {
        if (e.key === "ArrowDown") {
          e.preventDefault();
          openNavMenu(g);
        } else if (e.key === "Escape" && !g.menu.hidden) {
          e.preventDefault();
          closeNavMenus();
        }
      });
      g.menu.addEventListener("click", (e) => {
        if (!e.target.closest("[data-view]")) return;
        closeNavMenus();
        g.trigger.focus();
      });
      g.menu.addEventListener("keydown", (e) => {
        const items = Array.from(g.menu.querySelectorAll('[role="menuitem"]'));
        const i = items.indexOf(document.activeElement);
        if (e.key === "Escape") {
          e.preventDefault();
          closeNavMenus();
          g.trigger.focus();
        } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
          e.preventDefault();
          if (!items.length) return;
          const step = e.key === "ArrowDown" ? 1 : -1;
          items[(i + step + items.length) % items.length].focus();
        } else if (e.key === "Home" || e.key === "End") {
          e.preventDefault();
          items[e.key === "Home" ? 0 : items.length - 1].focus();
        } else if (e.key === "Tab") {
          closeNavMenus();
          g.trigger.focus();
        }
      });
    });

    document.addEventListener("click", (e) => {
      if (!e.target.closest || !e.target.closest("#tabs .nav-group")) closeNavMenus();
    });
    document.addEventListener("scroll", (e) => {
      if (e.target.closest && e.target.closest(".nav-menu")) return;
      closeNavMenus();
    }, true);
    window.addEventListener("resize", () => closeNavMenus());

    el.prev.addEventListener("click", () => navStep(-1));
    el.next.addEventListener("click", () => navStep(1));

    document.addEventListener("keydown", (e) => {
      if (e.target.closest && e.target.closest(".nav-group")) return;
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if (DOC_CHAPTERS.includes(state.view)) {
        if (stageOf(state.view) !== "explore") return;
        if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
          e.preventDefault();
          scrollDocTo(state.view, chapterStepIndex(state.view) + (e.key === "ArrowLeft" ? -1 : 1));
        }
        return;
      }
      if (!CHAPTERS.includes(state.view) || stageOf(state.view) !== "explore") return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        navStep(-1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        navStep(1);
      } else if (e.key === " ") {
        if (tag === "button") return;
        e.preventDefault();
        togglePlay();
      }
    });

    document.addEventListener("lk:navigate", (e) => {
      const detail = e.detail || {};
      if (!VIEWS.includes(detail.view)) return;
      if (CHAPTERS.includes(detail.view) || DOC_CHAPTERS.includes(detail.view)) {
        // 旧链接/题库里用 summary 表示「反思与总结」，路由统一成 reflect。
        const stage = detail.stage === "summary" ? "reflect" : detail.stage;
        if (stage && STAGE_NAMES.includes(stage)) {
          state.stages[detail.view] = stage;
        } else if (typeof detail.step === "number") {
          state.stages[detail.view] = "explore";
          setChapterStep(detail.view, detail.step);
        }
      }
      setView(detail.view);
      updateHash();
    });

    const goKv = $("#go-kv");
    if (goKv) {
      goKv.addEventListener("click", () => {
        state.step = 0;
        setView("kv");
        updateHash();
      });
    }
    const goReuse = $("#go-reuse");
    if (goReuse) {
      goReuse.addEventListener("click", () => {
        state.stages.prefix = "explore";
        setChapterStep("prefix", 10);
        setView("prefix");
        updateHash();
      });
    }

    window.addEventListener("hashchange", route);
  }

  /* ------------------------------------------------------- code drawer */

  function initCodeDrawers() {
    const views = [el.kv, el.prefix, el.paged];
    let mobile = window.innerWidth <= 980;
    const apply = (chapter, btn, collapsed) => {
      chapter.classList.toggle("code-collapsed", collapsed);
      btn.setAttribute("aria-expanded", String(!collapsed));
      btn.textContent = collapsed ? "展开代码 ▾" : "收起代码 ▴";
    };
    const entries = views
      .map((view) => {
        if (!view) return null;
        const chapter = view.querySelector(".chapter");
        const btn = view.querySelector(".code-toggle");
        return chapter && btn ? { chapter, btn } : null;
      })
      .filter(Boolean);
    entries.forEach(({ chapter, btn }) => {
      apply(chapter, btn, mobile);
      btn.addEventListener("click", () => apply(chapter, btn, !chapter.classList.contains("code-collapsed")));
    });
    window.addEventListener("resize", () => {
      const now = window.innerWidth <= 980;
      if (now === mobile) return;
      mobile = now;
      entries.forEach(({ chapter, btn }) => apply(chapter, btn, now));
    });
  }

  /* ------------------------------------------------------ next buttons */

  function nextChapterId(ch) {
    const i = LEARNING_ORDER.indexOf(ch);
    return i >= 0 ? LEARNING_ORDER[i + 1] || null : null;
  }

  function stepNextHost(ch) {
    return ch === "kv" ? el.kvStepNext : ch === "prefix" ? el.prefixStepNext : el.pagedStepNext;
  }

  function renderStepNext(ch) {
    const host = stepNextHost(ch);
    if (!host) return;
    if (stageOf(ch) !== "explore") {
      host.innerHTML = "";
      return;
    }
    const steps = ch === "kv" ? kv.steps : ch === "prefix" ? LK_DATA.prefix.steps : LK_DATA.paged.steps;
    const idx = ch === "kv" ? state.step : ch === "prefix" ? state.prefix.step : state.paged.step;
    const last = idx >= steps.length - 1;
    const nextId = nextChapterId(ch);
    const label = last
      ? (nextId ? "下一章：" + (CHAPTER_LABELS[nextId] || nextId) : "学习反馈")
      : "下一个小节：" + steps[idx + 1].title;
    host.innerHTML = '<button type="button" class="btn ' + (last ? "primary" : "ghost") +
      ' step-next-btn" data-step-next="' + (last ? "chapter" : "step") + '">' +
      esc(label) + " →</button>";
    host.querySelector("button").addEventListener("click", () => {
      if (last) goNextChapter(ch);
      else navStep(1);
    });
  }

  /* ------------------------------------------------------ auto advance */

  function goNextChapter(ch) {
    const i = LEARNING_ORDER.indexOf(ch);
    if (i < 0) return;
    const next = LEARNING_ORDER[i + 1];
    if (!next) {
      // 最后一章读完进入学习反馈，形成闭环。
      setView("feedback");
      updateHash();
      return;
    }
    state.stages[next] = "think";
    setChapterStep(next, 0);
    setView(next);
    updateHash();
  }

  /* --------------------------------------------------------------- theme */

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    const btn = document.getElementById("theme-toggle");
    if (btn) {
      const light = theme === "light";
      btn.textContent = light ? "☾ 暗色" : "☀ 亮色";
      btn.setAttribute("aria-pressed", String(light));
    }
    try {
      localStorage.setItem("whykvc.theme", theme);
    } catch (e) {
      /* ignore */
    }
  }

  function initTheme() {
    let theme = null;
    try {
      theme = localStorage.getItem("whykvc.theme");
    } catch (e) {
      /* ignore */
    }
    applyTheme(theme || "dark");
    const btn = document.getElementById("theme-toggle");
    if (btn) btn.addEventListener("click", () => {
      applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
    });
  }

  /* ---------------------------------------------------------------- init */

  if (el.ovWhy && D.overview) {
    el.ovWhy.innerHTML = "<h2>" + esc(D.overview.whyTitle) + "</h2>" + D.overview.whyHTML;
  }
  initTheme();
  initStages();
  initNav();
  initCodeDrawers();
  LK.nextChapter = goNextChapter;
  buildDots();
  if (LK.sectionnav) {
    LK.sectionnav.mount($("#toc-calc"), el.calcBody);
  }
  codeView = LK.codeview.mount(el.code, kv.lines);
  prefixCode = LK.codeview.mount(el.prefixCode, LK_DATA.prefix.lines);
  pagedCode = LK.codeview.mount(el.pagedCode, LK_DATA.paged.lines);
  bind();
  route();
})();
