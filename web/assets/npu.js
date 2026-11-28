/* Document chapter: GPU / NPU / 国产算力的认知地图。
 * Content lives in web/tools/npu.json; example/npu_sim.py is injected by build.py.
 * One interactive: filter the vendor cards by GPU / NPU / domestic. */
(function () {
  const LK = (window.LK = window.LK || {});
  const N = (LK.npu = LK.npu || {});
  const esc = (s) => String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const arr = (x) => (Array.isArray(x) ? x : []);

  function sectionHead(sec) {
    sec = sec || {};
    return '<h2 class="sec-title">' + esc(sec.title) + "</h2>" +
      (sec.lead ? '<p class="lead">' + esc(sec.lead) + "</p>" : "");
  }

  function whyHTML(sec) {
    return sectionHead(sec) +
      '<div class="npu-grid">' +
      arr(sec.cards).map((c, i) =>
        '<div class="npu-card"><div class="npu-card-head"><span class="npu-no">0' + (i + 1) + "</span><b>" +
        esc(c.name) + '</b><span class="npu-tag">' + esc(c.tag) + "</span></div><p>" + esc(c.does) + "</p></div>"
      ).join("") +
      "</div>" + (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "");
  }

  function compareHTML(sec) {
    return sectionHead(sec) +
      '<div class="table-wrap"><table class="tagtable"><thead><tr>' +
      "<th>维度</th><th>GPU</th><th>NPU</th><th>ASIC / TPU</th><th>说明</th></tr></thead><tbody>" +
      arr(sec.rows).map((r) =>
        "<tr><td><b>" + esc(r.dim) + "</b></td><td>" + esc(r.gpu) + "</td><td>" + esc(r.npu) +
        "</td><td>" + esc(r.asic) + "</td><td>" + esc(r.note) + "</td></tr>"
      ).join("") +
      "</tbody></table></div>" + (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "");
  }

  function vendorHTML(sec) {
    return sectionHead(sec) +
      '<div class="npu-filters">' +
      arr(sec.filters).map((f, i) =>
        '<button type="button" class="npu-filter' + (i === 0 ? " on" : "") + '" data-npu-filter="' +
        esc(f.id) + '">' + esc(f.label) + "</button>"
      ).join("") +
      "</div>" +
      '<div class="npu-grid" id="npu-vendors">' +
      arr(sec.cards).map((c) =>
        '<div class="npu-card npu-vendor" data-kind="' + esc(c.kind) + '" data-region="' + esc(c.region) + '">' +
        '<div class="npu-card-head"><b>' + esc(c.name) + '</b><span class="npu-tag">' +
        esc(c.region === "cn" ? "国产" : "国际") + " · " + esc(c.kind.toUpperCase()) + "</span></div>" +
        '<p class="npu-vendor-meta">' + esc(c.arch) + " · " + esc(c.stack) + "</p>" +
        '<p class="npu-vendor-note">' + esc(c.note) + "</p></div>"
      ).join("") +
      "</div>" + (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "");
  }

  function stackHTML(sec) {
    return sectionHead(sec) +
      '<div class="table-wrap"><table class="tagtable"><thead><tr>' +
      "<th>厂商</th><th>硬件</th><th>软件栈</th><th>底层</th><th>备注</th></tr></thead><tbody>" +
      arr(sec.rows).map((r) =>
        "<tr><td><b>" + esc(r.vendor) + "</b></td><td>" + esc(r.hardware) + "</td><td>" +
        esc(r.stack) + "</td><td>" + esc(r.base) + "</td><td>" + esc(r.note) + "</td></tr>"
      ).join("") +
      "</tbody></table></div>" + (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "");
  }

  function llmHTML(sec) {
    return sectionHead(sec) +
      '<div class="table-wrap"><table class="tagtable"><thead><tr>' +
      "<th>阶段</th><th>瓶颈</th><th>更适合</th><th>为什么</th></tr></thead><tbody>" +
      arr(sec.rows).map((r) =>
        "<tr><td><b>" + esc(r.phase) + "</b></td><td>" + esc(r.bottleneck) + "</td><td>" +
        esc(r.prefer) + "</td><td>" + esc(r.why) + "</td></tr>"
      ).join("") +
      "</tbody></table></div>" + (sec.note ? '<p class="note">' + esc(sec.note) + "</p>" : "");
  }

  function selectHTML(sec) {
    return sectionHead(sec) +
      '<div class="npu-grid">' +
      arr(sec.items).map((it, i) =>
        '<div class="npu-card"><div class="npu-card-head"><span class="npu-no">0' + (i + 1) + "</span><b>" +
        esc(it.scene) + "</b></div><p><strong>" + esc(it.choice) + "</strong></p><p>" + esc(it.why) + "</p></div>"
      ).join("") +
      "</div>" + (sec.conclusion ? '<p class="npu-conclusion">' + esc(sec.conclusion) + "</p>" : "");
  }

  function codeHTML(sec, simOutput) {
    return sectionHead(sec) +
      '<pre class="train-run"><code>' + esc(sec.run) + "</code></pre>" +
      '<ul class="train-rules">' + arr(sec.notes).map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul>" +
      '<div class="train-code-head"><span>' + esc(sec.file) + "</span><span>纯 Python · 可运行</span></div>" +
      '<div class="train-code" id="npu-sim-code"></div>' +
      '<h3 class="sec-sub">示例输出</h3>' +
      '<pre class="train-run"><code>' + esc(simOutput) + "</code></pre>";
  }

  function checkHTML(sec) {
    return sectionHead(sec) +
      '<ul class="train-rules train-checklist">' +
      arr(sec.checklist).map((c) => "<li>" + esc(c) + "</li>").join("") + "</ul>" +
      arr(sec.questions).map((q) =>
        '<details class="thinkbox"><summary>' + esc(q.q) + "</summary><p>" + esc(q.a) + "</p></details>"
      ).join("");
  }

  N.render = function (el, data) {
    if (!el) return;
    data = data || {};
    el.innerHTML =
      '<p class="kicker">' + esc(data.kicker) + "</p>" +
      '<h1 class="page-title">' + esc(data.title) + "</h1>" +
      '<p class="page-lead">' + esc(data.lead) + "</p>" +
      whyHTML(data.why) + compareHTML(data.compare) + vendorHTML(data.vendors) +
      stackHTML(data.stack) + llmHTML(data.llm) + selectHTML(data.select) +
      codeHTML(data.codeSection, data.simOutput) + checkHTML(data.check);

    const host = el.querySelector("#npu-vendors");
    el.querySelectorAll("[data-npu-filter]").forEach((btn) =>
      btn.addEventListener("click", () => {
        const f = btn.dataset.npuFilter;
        el.querySelectorAll("[data-npu-filter]").forEach((b) => b.classList.toggle("on", b === btn));
        if (!host) return;
        host.querySelectorAll(".npu-vendor").forEach((card) => {
          const show = f === "all" || card.dataset.kind === f || card.dataset.region === f;
          card.hidden = !show;
        });
      })
    );

    const codeEl = el.querySelector("#npu-sim-code");
    if (codeEl && LK.codeview && data.simCode) LK.codeview.mount(codeEl, data.simCode.lines);
  };
})();
