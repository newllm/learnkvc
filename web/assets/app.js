/* App shell: thinking-first navigation + chapter stepper + playback. */
(function () {
  const LK = (window.LK = window.LK || {});
  const D = window.LK_DATA;
  const TR = window.LK_TRACES && window.LK_TRACES.kv;

  if (!D || !TR) {
    document.body.innerHTML =
      '<p style="padding:2rem;color:#d7e2ee">数据缺失：先运行 <code>python3 web/tools/build.py</code>。</p>';
    return;
  }

  const kv = D.kv;
  const MAX_GEN = TR.generated.length;
  const VIEWS = ["think", "kv", "prefix", "paged", "anatomy", "calc", "feedback", "homework"];
  const state = { view: "think", step: 0, genStep: MAX_GEN, playing: false, timer: null };

  const $ = (sel) => document.querySelector(sel);
  const el = {
    think: $("#view-think"),
    kv: $("#view-kv"),
    prefix: $("#view-prefix"),
    paged: $("#view-paged"),
    anatomy: $("#view-anatomy"),
    calc: $("#view-calc"),
    feedback: $("#view-feedback"),
    homework: $("#view-homework"),
    tabs: Array.from(document.querySelectorAll("#tabs button")),
    code: $("#code"),
    codeMeta: $("#code-meta"),
    prefixCode: $("#prefix-code"),
    prefixCodeMeta: $("#prefix-code-meta"),
    prefixDoc: $("#prefix-doc"),
    prefixViz: $("#prefix-viz"),
    pagedCode: $("#paged-code"),
    pagedCodeMeta: $("#paged-code-meta"),
    pagedDoc: $("#paged-doc"),
    pagedViz: $("#paged-viz"),
    stepNavs: { kv: $("#stepnav-kv"), prefix: $("#stepnav-prefix"), paged: $("#stepnav-paged") },
    homeworkBody: $("#homework"),
    doc: $("#doc-card"),
    viz: $("#viz"),
    dots: $("#dots"),
    counters: $("#counters"),
    prev: $("#prev"),
    next: $("#next"),
    quiz: $("#quiz"),
    anatomyBody: $("#anatomy"),
    calcBody: $("#calc"),
    feedbackBody: $("#feedback"),
    ovWhy: $("#ov-why"),
  };

  let codeView = null;
  let prefixCode = null;
  let pagedCode = null;

  state.prefix = { step: 0, scenario: "shared", snap: 0, playing: false, timer: null };
  state.paged = { step: 0, snap: 0, sel: "seq-b", playing: false, timer: null };
  const CHAPTERS = ["kv", "prefix", "paged"];
  const STAGE_NAMES = ["think", "explore", "reflect", "summary"];
  state.stages = {};
  state.navCollapsed = { kv: false, prefix: false, paged: false };

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

  function renderStep() {
    const step = kv.steps[state.step];
    if (LK.progress) LK.progress.recordStep(step.id);
    renderDoc(step, state.step);
    const right = document.getElementById("right-pane");
    if (right) right.scrollTop = 0;
    if (codeView) {
      codeView.highlight(step.codeRanges);
      if (step.codeRanges.length) codeView.scrollToLine(step.codeRanges[0].start);
      el.codeMeta.textContent = step.codeRanges.length
        ? "高亮 " + step.codeRanges.map((r) => r.start + "–" + r.end).join(" · ") + " 行"
        : "讲解步骤";
    }
    renderViz();
    renderCounters();
    updateDots();
    renderChapterNav("kv");
    if (window.LK && LK.progress) {
      const l = LK.progress.lesson("kv");
      if ((l.exploreMax || 0) < state.step + 1) LK.progress.saveLesson("kv", { exploreMax: state.step + 1 });
      renderChapterNav("kv");
    }
  }

  function setStep(i) {
    if (state.playing) stopPlay(false);
    state.step = clamp(i);
    renderStep();
    updateHash();
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
    const right = document.getElementById("prefix-right");
    if (right) right.scrollTop = 0;
    if (prefixCode) {
      prefixCode.highlight(step.codeRanges);
      if (step.codeRanges.length) prefixCode.scrollToLine(step.codeRanges[0].start);
      el.prefixCodeMeta.textContent = step.codeRanges.length
        ? "高亮 " + step.codeRanges.map((r) => r.start + "–" + r.end).join(" · ") + " 行"
        : "讲解步骤";
    }
    renderPrefixViz();
    renderPrefixCounters();
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
    stopPrefixPlay();
    state.prefix.step = Math.max(0, Math.min(LK_DATA.prefix.steps.length - 1, i));
    renderPrefixStep();
    updateHash();
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
    const right = document.getElementById("paged-right");
    if (right) right.scrollTop = 0;
    if (pagedCode) {
      pagedCode.highlight(step.codeRanges);
      if (step.codeRanges.length) pagedCode.scrollToLine(step.codeRanges[0].start);
      el.pagedCodeMeta.textContent = step.codeRanges.length
        ? "高亮 " + step.codeRanges.map((r) => r.start + "–" + r.end).join(" · ") + " 行"
        : "讲解步骤";
    }
    renderPagedViz();
    renderPagedCounters();
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
    stopPagedPlay();
    state.paged.step = Math.max(0, Math.min(LK_DATA.paged.steps.length - 1, i));
    renderPagedStep();
    updateHash();
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
      return JSON.parse(localStorage.getItem("learnkvc.stepnav") || "{}") || {};
    } catch (e) {
      return {};
    }
  }

  function saveNavStore() {
    try {
      localStorage.setItem("learnkvc.stepnav", JSON.stringify(state.navCollapsed));
    } catch (e) {
      /* localStorage unavailable */
    }
  }

  function initNav() {
    const saved = navStore();
    CHAPTERS.forEach((ch) => {
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

    const stageDefs = [
      ["think", "请回答", lesson.think ? "✓" : ""],
      ["explore", "探索阅读", (lesson.exploreMax || 0) + " / " + steps.length],
      ["reflect", "反思与归因", lesson.reflect && lesson.attributionSubmitted ? "✓" : lesson.reflect ? "待归因" : ""],
      ["summary", "总结归纳", lesson.summary ? "✓" : ""],
    ];
    const stagesHTML = stageDefs
      .map(
        ([id, label, sub], i) =>
          '<button class="stage-nav-item' + (id === stage ? " on" : "") +
          (sub === "✓" ? " done" : "") + '" data-nav-stage="' + id + '">' +
          '<span class="sid">0' + (i + 1) + "</span>" +
          '<span class="stt">' + label + "</span>" +
          '<em>' + sub + "</em></button>"
      )
      .join("");
    const showSteps = stage === "explore";
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
        if (ch === "kv") setStep(i);
        else if (ch === "prefix") setPrefixStep(i);
        else setPagedStep(i);
      })
    );
    applyNavCollapsed(ch);
    const list = nav.querySelector(".stepnav-list");
    const active = list && list.querySelector(".active");
    if (list && active) {
      list.scrollTop = Math.max(0, active.offsetTop - list.clientHeight / 2);
    }
  }

  /* ------------------------------------------------------- lesson stages */

  function chapterSteps(ch) {
    return ch === "kv" ? kv.steps : ch === "prefix" ? LK_DATA.prefix.steps : LK_DATA.paged.steps;
  }

  function chapterStepIndex(ch) {
    return ch === "kv" ? state.step : ch === "prefix" ? state.prefix.step : state.paged.step;
  }

  function setChapterStep(ch, i) {
    const max = chapterSteps(ch).length - 1;
    const v = Math.max(0, Math.min(max, i));
    if (ch === "kv") state.step = v;
    else if (ch === "prefix") state.prefix.step = v;
    else state.paged.step = v;
  }

  function stageOf(ch) {
    return state.stages[ch] || "think";
  }

  function initStages() {
    CHAPTERS.forEach((ch) => {
      const l = LK.progress.lesson(ch);
      state.stages[ch] = l.summary ? "summary" : l.reflect ? "summary" : l.think ? "explore" : "think";
    });
  }

  function setStage(ch, stage) {
    if (!STAGE_NAMES.includes(stage)) return;
    state.stages[ch] = stage;
    if (stage !== "explore") {
      if (ch === "prefix") stopPrefixPlay();
      if (ch === "paged") stopPagedPlay();
    }
    renderChapterStage(ch);
    updateHash();
  }

  function renderChapterStage(ch) {
    const view = el[ch];
    if (!view) return;
    STAGE_NAMES.forEach((s) => view.classList.toggle("stage-" + s, stageOf(ch) === s));
    renderChapterNav(ch);
    const timeline = document.querySelector(".timeline");
    const docEl = ch === "kv" ? el.doc : ch === "prefix" ? el.prefixDoc : el.pagedDoc;
    const vizEl = ch === "kv" ? el.viz : ch === "prefix" ? el.prefixViz : el.pagedViz;

    if (stageOf(ch) === "explore") {
      if (timeline) timeline.style.display = "flex";
      if (ch === "kv") renderStep();
      else if (ch === "prefix") renderPrefixStep();
      else renderPagedStep();
      return;
    }
    if (timeline) timeline.style.display = "none";
    if (vizEl) vizEl.innerHTML = "";
    const lesson = LK_DATA.lessons[ch];
    if (stageOf(ch) === "think") {
      LK.workshop.renderStageThink(docEl, ch, lesson.think, () => renderChapterNav(ch), () =>
        setStage(ch, "explore")
      );
    } else if (stageOf(ch) === "reflect") {
      LK.workshop.renderStageReflect(docEl, ch, lesson.reflect, lesson.attribution, () => renderChapterNav(ch), () =>
        setStage(ch, "summary")
      );
    } else {
      LK.workshop.renderStageSummary(docEl, ch, lesson.summary, () => renderChapterNav(ch), () => {
        setView("homework");
        updateHash();
      });
    }
  }

  /* ---------------------------------------------------------- dots/hash */

  function navStep(delta) {
    if (!CHAPTERS.includes(state.view) || stageOf(state.view) !== "explore") return;
    if (state.view === "kv") setStep(state.step + delta);
    else if (state.view === "prefix") setPrefixStep(state.prefix.step + delta);
    else if (state.view === "paged") setPagedStep(state.paged.step + delta);
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
    let hash = "#" + state.view;
    if (CHAPTERS.includes(state.view)) {
      const ch = state.view;
      const stage = stageOf(ch);
      hash = stage === "explore" ? "#" + ch + "/" + (chapterStepIndex(ch) + 1) : "#" + ch + "/" + stage;
    }
    if (location.hash !== hash) history.replaceState(null, "", hash);
  }

  /* -------------------------------------------------------------- views */

  function setView(view) {
    if (state.playing) stopPlay(false);
    state.view = view;
    document.body.dataset.view = view;
    const timeline = document.querySelector(".timeline");
    if (timeline) timeline.style.display = "";
    if (LK.progress) LK.progress.recordView(view);
    el.tabs.forEach((t) => t.classList.toggle("active", t.dataset.view === view));
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
    } else if (view === "anatomy") {
      LK.anatomy.render(el.anatomyBody, D.anatomy);
    } else if (view === "calc") {
      LK.calc.render(el.calcBody, {
        presets: D.anatomy.presets,
        context: D.anatomy.context,
      });
    } else if (view === "homework") {
      LK.workshop.renderHomework(el.homeworkBody, D.homework, (ch) => {
        setView(ch);
        updateHash();
      });
    } else if (view === "feedback") {
      LK.workshop.renderFeedback(el.feedbackBody, {
        preQuiz: D.think.quiz,
        postTest: D.think.postTest,
        takeaway: D.reflect.takeaway,
        summary: D.reflect.feedbackSummary,
        totalSteps: kv.steps.length + D.prefix.steps.length + D.paged.steps.length,
      });
    }
  }

  function route() {
    const hash = location.hash.replace(/^#/, "");
    for (const ch of CHAPTERS) {
      if (hash === ch || hash.startsWith(ch + "/")) {
        const rest = hash.slice(ch.length + 1);
        if (STAGE_NAMES.includes(rest)) {
          state.stages[ch] = rest;
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
      state.stages.kv = "summary";
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
        setView(tab.dataset.view);
        updateHash();
      });
    });

    el.prev.addEventListener("click", () => navStep(-1));
    el.next.addEventListener("click", () => navStep(1));

    document.addEventListener("keydown", (e) => {
      if (!CHAPTERS.includes(state.view) || stageOf(state.view) !== "explore") return;
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
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
      if (CHAPTERS.includes(detail.view)) {
        if (detail.stage && STAGE_NAMES.includes(detail.stage)) {
          state.stages[detail.view] = detail.stage;
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
      localStorage.setItem("learnkvc.theme", theme);
    } catch (e) {
      /* ignore */
    }
  }

  function initTheme() {
    let theme = null;
    try {
      theme = localStorage.getItem("learnkvc.theme");
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
  buildDots();
  codeView = LK.codeview.mount(el.code, kv.lines);
  prefixCode = LK.codeview.mount(el.prefixCode, LK_DATA.prefix.lines);
  pagedCode = LK.codeview.mount(el.pagedCode, LK_DATA.paged.lines);
  bind();
  route();
})();
