/* Prefix cache chapter: request timeline + block pool + hash chain + LRU. */
(function () {
  const LK = (window.LK = window.LK || {});
  const V = (LK.vizPrefix = LK.vizPrefix || {});
  const fmt = (n) => Number(n).toLocaleString("en-US");
  const esc = (s) =>
    String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const SCENARIOS = [
    ["shared", "共享 system prompt"],
    ["eviction", "池子不足 · LRU"],
    ["repeated", "相同 token 不同位置"],
  ];

  function stat(label, value, sub) {
    return (
      '<div class="stat"><div class="stat-label">' + esc(label) + "</div>" +
      '<div class="stat-value">' + value + "</div>" +
      '<div class="stat-sub">' + esc(sub) + "</div></div>"
    );
  }

  V.render = function (el, step, ctx) {
    const tr = ctx.trace;
    const sc = ctx.state.scenario;
    const data = tr.scenarios[sc] || tr.scenarios.shared;
    const snaps = data.snapshots;
    const idx = Math.max(0, Math.min(snaps.length - 1, ctx.state.snap));
    const snap = snaps[idx];
    const prev = idx > 0 ? snaps[idx - 1] : null;
    const prevIds = new Set((prev ? prev.blocks : []).map((b) => b.id));
    const blocks = snap.blocks;
    const byHash = {};
    blocks.forEach((b) => (byHash[b.hash] = b));

    const used = blocks.length;
    const pinned = blocks.filter((b) => b.ref > 0).length;
    const newIds = new Set(blocks.filter((b) => !prevIds.has(b.id)).map((b) => b.id));

    // hash chain of the newest block
    let chain = [];
    let cursor = blocks.length ? blocks[blocks.length - 1] : null;
    let guard = 0;
    while (cursor && guard++ < 30) {
      chain.unshift(cursor);
      cursor = cursor.parent ? byHash[cursor.parent] : null;
    }

    const requestRows = data.requests
      .map(
        (r) =>
          "<tr" + (snap.active.includes(r.id) ? ' class="active"' : "") + ">" +
          "<td>" + esc(r.id) + "</td>" +
          "<td>" + r.prompt + "</td>" +
          "<td>" + (r.cached ? r.cached + " tok" : "—") + "</td>" +
          "<td>" + r.prefill + " tok</td>" +
          "<td>" + (r.blocks.length ? r.blocks.map(esc).join(" ") : "—") + "</td>" +
          "</tr>"
      )
      .join("");

    const pool = blocks
      .map((b) => {
        const cls =
          "pblock" +
          (b.ref > 0 ? " pinned" : " idle") +
          (newIds.has(b.id) ? " new" : "");
        return (
          '<button class="' + cls + '" data-block="' + b.id + '" title="' +
          esc("b" + b.id + " · hash " + b.hash + " · ref " + b.ref + " · tokens " + b.tokens.join(",")) + '">' +
          '<b>b' + b.id + "</b><span>" + esc(b.hash) + "</span>" +
          '<em>ref ' + b.ref + "</em></button>"
        );
      })
      .join("");

    el.innerHTML =
      '<div class="viz-head"><h3>块池 · 请求时间线</h3>' +
      '<span class="sub">' + esc(data.title) + " · capacity " + data.capacity + " 块</span></div>" +
      '<div class="mode-tabs">' +
      SCENARIOS.map(
        ([id, label]) =>
          '<button data-scenario="' + id + '" class="' + (id === sc ? "on" : "") + '">' +
          esc(label) + "</button>"
      ).join("") +
      "</div>" +
      '<div class="reuse-toolbar"><div class="controls">' +
      '<button class="btn ghost" data-nav="-1">←</button>' +
      '<button class="btn" data-play>' + (ctx.playing ? "⏸ 暂停" : "▶ 播放") + "</button>" +
      '<button class="btn ghost" data-nav="1">→</button>' +
      '<input type="range" class="snaprange" data-snap min="0" max="' + (snaps.length - 1) +
      '" value="' + idx + '">' +
      '<span class="tool-note">t=' + snap.step + " · " + esc(snap.phase) +
      " · clock " + snap.clock + " · " + (idx + 1) + "/" + snaps.length + "</span>" +
      "</div></div>" +
      '<div class="compare">' +
      stat("命中率", Math.round(data.stats.hit_rate * 100) + "%",
        fmt(data.stats.reused_tokens) + " / " + fmt(data.stats.prompt_tokens) + " tok 复用") +
      stat("本场景 prefill", fmt(data.stats.prefilled_tokens) + " tok",
        "命中多少就少算多少") +
      stat("池占用", used + " / " + data.capacity,
        "占用 " + pinned + " 块 · 淘汰 " + data.stats.evicted) +
      "</div>" +
      '<div class="table-wrap"><table class="reqtable"><thead><tr>' +
      "<th>请求</th><th>prompt</th><th>复用</th><th>重算</th><th>持有的块</th>" +
      "</tr></thead><tbody>" + requestRows + "</tbody></table></div>" +
      '<h4 class="mini-title">物理块池（点击块看哈希链）</h4>' +
      '<div class="blockpool" style="grid-template-columns: repeat(auto-fill, minmax(58px, 1fr))">' + pool + "</div>" +
      '<div class="blockdetail" id="blockdetail">' + chainHTML(chain) + "</div>" +
      '<h4 class="mini-title">时间线</h4>' +
      '<div class="tracepts">' +
      snap.trace.map((line) => "<div>" + esc(line) + "</div>").join("") +
      "</div>" +
      '<p class="note">灰 = 空闲但仍在池里（可复用）；亮 = 被请求占着（ref&gt;0）；黄 = 本步新分配的块。' +
      "淘汰只发生在「没有引用、且没有孩子」的叶子上。</p>";

    el.querySelectorAll("[data-scenario]").forEach((b) =>
      b.addEventListener("click", () => ctx.setScenario(b.dataset.scenario))
    );
    el.querySelectorAll("[data-nav]").forEach((b) =>
      b.addEventListener("click", () => ctx.setSnap(idx + +b.dataset.nav))
    );
    const play = el.querySelector("[data-play]");
    if (play) play.addEventListener("click", () => ctx.togglePlay());
    const range = el.querySelector("[data-snap]");
    if (range) range.addEventListener("change", () => ctx.setSnap(+range.value));

    const detail = el.querySelector("#blockdetail");
    el.querySelectorAll("[data-block]").forEach((b) =>
      b.addEventListener("click", () => {
        const block = blocks.find((x) => x.id === +b.dataset.block);
        if (!block) return;
        let chain = [];
        let cur = block;
        let n = 0;
        while (cur && n++ < 30) {
          chain.unshift(cur);
          cur = cur.parent ? byHash[cur.parent] : null;
        }
        detail.innerHTML = chainHTML(chain);
      })
    );
  };

  function chainHTML(chain) {
    if (!chain || !chain.length) return '<span class="empty">还没有块</span>';
    return (
      '<span class="tool-note">哈希链（父 → 子）：</span>' +
      chain
        .map((b) => '<span class="miniblock">b' + b.id + " · " + esc(b.hash) + " · " + b.tokens.length + " tok</span>")
        .join('<span class="chainarrow">→</span>')
    );
  }
})();
