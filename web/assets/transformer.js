/* Document chapter: Transformer / LLM background and why it won.
 * Content lives in web/tools/transformer.json; example/transformer_sim.py is injected.
 * One interactive: sequence-length slider for attention O(N²), KV linear, FLOPs. */
(function () {
  const LK = (window.LK = window.LK || {});
  const T = (LK.transformer = LK.transformer || {});
  const esc = (s) => String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const arr = (x) => (Array.isArray(x) ? x : []);
  const LAYERS = 32, HEADS = 32, HEAD_DIM = 128, KV_HEADS = 8, BYTES = 2;

  const head = (sec) => '<h2 class="sec-title">' + esc(sec.title) + "</h2>" +
    (sec.lead ? '<p class="lead">' + esc(sec.lead) + "</p>" : "");

  function whyHTML(sec) {
    return head(sec) + '<div class="tf-grid">' + arr(sec.cards).map((c, i) =>
      '<div class="tf-card"><div class="tf-card-head"><span class="tf-no">0' + (i + 1) + "</span><b>" + esc(c.name) +
      '</b><span class="tf-tag">' + esc(c.tag) + "</span></div><p>" + esc(c.does) + "</p></div>"
    ).join("") + "</div>" + (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "");
  }

  function attnHTML(sec) {
    return head(sec) +
      '<ul class="train-rules">' + arr(sec.points).map((p) => "<li>" + esc(p) + "</li>").join("") + "</ul>" +
      '<div class="tf-lab"><div class="tf-lab-row"><label for="tf-seq">序列长度 N</label>' +
      '<input type="range" id="tf-seq" min="1024" max="131072" step="1024" value="4096">' +
      '<b id="tf-seq-val">4096</b></div>' +
      '<div class="tf-lab-grid" id="tf-nums"></div>' +
      '<p class="note">拖动看 O(N²) 的注意力与线性 KV：长上下文不是「免费塞更多」。</p></div>' +
      (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "");
  }

  function archHTML(sec) {
    return head(sec) +
      '<div class="table-wrap"><table class="tagtable"><thead><tr><th>部件</th><th>作用</th><th>要点</th></tr></thead><tbody>' +
      arr(sec.rows).map((r) => "<tr><td><b>" + esc(r.part) + "</b></td><td>" + esc(r.role) + "</td><td>" + esc(r.note) + "</td></tr>").join("") +
      "</tbody></table></div>" +
      '<div class="tf-grid tf-grid-3">' + arr(sec.arches).map((a) =>
        '<div class="tf-card"><b>' + esc(a.name) + "</b><p>" + esc(a.use) + "</p><p class=\"tf-meta\">" + esc(a.note) + "</p></div>"
      ).join("") + "</div>" + (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "");
  }

  function timelineHTML(sec) {
    return head(sec) + '<div class="tf-timeline">' + arr(sec.events).map((e) =>
      '<div class="tf-event"><span class="tf-year">' + esc(e.year) + "</span><b>" + esc(e.name) +
      "</b><p>" + esc(e.note) + "</p></div>"
    ).join("") + "</div>" + (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "");
  }

  function scalingHTML(sec) {
    return head(sec) +
      '<div class="table-wrap"><table class="tagtable"><thead><tr><th>轴</th><th>带来的收益</th><th>代价 / 注意</th></tr></thead><tbody>' +
      arr(sec.rows).map((r) => "<tr><td><b>" + esc(r.axis) + "</b></td><td>" + esc(r.effect) + "</td><td>" + esc(r.caution) + "</td></tr>").join("") +
      "</tbody></table></div>" + (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "");
  }

  function winHTML(sec) {
    return head(sec) +
      '<div class="tf-grid">' + arr(sec.reasons).map((r, i) =>
        '<div class="tf-card"><div class="tf-card-head"><span class="tf-no">0' + (i + 1) + "</span><b>" + esc(r.name) + "</b></div><p>" + esc(r.detail) + "</p></div>"
      ).join("") + "</div>" +
      '<h3 class="sec-sub">另一个视角：其他 AI 领域现在在哪</h3>' +
      '<div class="table-wrap"><table class="tagtable"><thead><tr><th>领域</th><th>过去</th><th>现在</th></tr></thead><tbody>' +
      arr(sec.contrast).map((r) => "<tr><td><b>" + esc(r.field) + "</b></td><td>" + esc(r.before) + "</td><td>" + esc(r.now) + "</td></tr>").join("") +
      "</tbody></table></div>" +
      '<p class="tf-conclusion">' + esc(sec.conclusion) + "</p>";
  }

  function limitsHTML(sec) {
    return head(sec) + '<div class="tf-grid">' + arr(sec.items).map((it) =>
      '<div class="tf-card"><b>' + esc(it.name) + "</b><p>" + esc(it.note) + "</p></div>"
    ).join("") + "</div>" + (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "");
  }

  function codeHTML(sec, simOutput) {
    return head(sec) +
      '<pre class="train-run"><code>' + esc(sec.run) + "</code></pre>" +
      '<ul class="train-rules">' + arr(sec.notes).map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul>" +
      '<div class="train-code-head"><span>' + esc(sec.file) + "</span><span>纯 Python · 可运行</span></div>" +
      '<div class="train-code" id="tf-sim-code"></div>' +
      '<h3 class="sec-sub">示例输出</h3><pre class="train-run"><code>' + esc(simOutput) + "</code></pre>";
  }

  function checkHTML(sec) {
    return head(sec) +
      '<ul class="train-rules train-checklist">' + arr(sec.checklist).map((c) => "<li>" + esc(c) + "</li>").join("") + "</ul>" +
      arr(sec.questions).map((q) => '<details class="thinkbox"><summary>' + esc(q.q) + "</summary><p>" + esc(q.a) + "</p></details>").join("");
  }

  function mountLab(root) {
    const input = root.querySelector("#tf-seq");
    const val = root.querySelector("#tf-seq-val");
    const nums = root.querySelector("#tf-nums");
    const draw = () => {
      const n = +input.value;
      const scores = HEADS * n * n;
      const tflops = 4 * LAYERS * HEADS * n * n * HEAD_DIM / 1e12;
      const kv = (2 * LAYERS * KV_HEADS * HEAD_DIM * n * BYTES) / 2 ** 30;
      val.textContent = n.toLocaleString();
      nums.innerHTML =
        '<div><span>注意力分数元素</span><b>' + scores.toLocaleString() + "</b></div>" +
        '<div><span>attention TFLOPs</span><b>' + tflops.toFixed(1) + "</b></div>" +
        '<div><span>KV cache</span><b>' + kv.toFixed(1) + " GiB</b></div>";
    };
    input.addEventListener("input", draw);
    draw();
  }

  T.render = function (el, data) {
    if (!el) return;
    data = data || {};
    el.innerHTML =
      '<p class="kicker">' + esc(data.kicker) + "</p>" +
      '<h1 class="page-title">' + esc(data.title) + "</h1>" +
      '<p class="page-lead">' + esc(data.lead) + "</p>" +
      whyHTML(data.why) + attnHTML(data.attn) + archHTML(data.arch) +
      timelineHTML(data.timeline) + scalingHTML(data.scaling) + winHTML(data.win) +
      limitsHTML(data.limits) + codeHTML(data.codeSection, data.simOutput) + checkHTML(data.check);

    mountLab(el);
    const codeEl = el.querySelector("#tf-sim-code");
    if (codeEl && LK.codeview && data.simCode) LK.codeview.mount(codeEl, data.simCode.lines);
  };
})();
