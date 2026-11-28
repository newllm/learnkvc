/* PagedAttention chapter: logical tokens -> block table -> physical pool. */
(function () {
  const LK = (window.LK = window.LK || {});
  const V = (LK.vizPaged = LK.vizPaged || {});
  const esc = (s) =>
    String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  V.render = function (el, step, ctx) {
    const tr = ctx.trace;
    const snaps = tr.snapshots;
    const idx = Math.max(0, Math.min(snaps.length - 1, ctx.state.snap));
    const snap = snaps[idx];
    const prev = idx > 0 ? snaps[idx - 1] : null;
    const sel = ctx.state.sel && tr.sequences.includes(ctx.state.sel) ? ctx.state.sel : "seq-b";
    const table = snap.tables[sel] || [];
    const length = snap.lengths[sel] || 0;
    const waste = table.length * tr.block_size - length;
    const used = snap.blocks.filter((b) => b.owner).length;
    const focus = ctx.state.block;
    const reused = snap.event === "reuse" ? new Set(tr.reused_by_c || []) : new Set();

    const prevOwners = {};
    if (prev) prev.blocks.forEach((b) => (prevOwners[b.id] = b.owner));
    const newBlocks = new Set(
      snap.blocks.filter((b) => b.owner && (prev ? prevOwners[b.id] !== b.owner : true)).map((b) => b.id)
    );

    const tokenChips = Array.from({ length }, (_, i) => {
      const blockIdx = Math.floor(i / tr.block_size);
      const slot = i % tr.block_size;
      const bid = table[blockIdx];
      return (
        '<span class="ptok" title="逻辑位置 ' + i + " → 块 b" + bid + " slot " + slot + '">' + i + "</span>"
      );
    }).join("");

    const seqCards = tr.sequences
      .map((s) => {
        const t = snap.tables[s] || [];
        const len = snap.lengths[s] || 0;
        if (!t.length) {
          return (
            '<div class="seqcard empty' + (s === sel ? " sel" : "") + '" data-seq="' + s + '">' +
            "<b>" + s + '</b><span class="tool-note">未分配 / 已释放</span></div>'
          );
        }
        const rows = t
          .map((bid, logical) => {
            const slots = Math.max(0, Math.min(tr.block_size, len - logical * tr.block_size));
            const block = snap.blocks.find((b) => b.id === bid);
            return (
              '<button class="btrow' + (bid === focus ? " focus" : "") +
              (reused.has(bid) ? " reused" : "") + '" data-block="' + bid + '">' +
              '<span class="btlogic">#' + logical + "</span>" +
              '<span class="btid">b' + bid + "</span>" +
              '<span class="btslots">' + slots + "/" + tr.block_size + " slot</span>" +
              "</button>"
            );
          })
          .join("");
        return (
          '<div class="seqcard' + (s === sel ? " sel" : "") + '" data-seq="' + s + '">' +
          '<div class="seqhead"><b>' + s + '</b><span class="tool-note">' +
          len + " tok · " + t.length + " 块 · 浪费 " + (t.length * tr.block_size - len) + " slot</span></div>" +
          '<div class="btable">' + rows + "</div></div>"
        );
      })
      .join("");

    const pool = snap.blocks
      .map((b) => {
        const cls =
          "pblock" +
          (b.owner ? " pinned" : " idle") +
          (newBlocks.has(b.id) ? " new" : "") +
          ((snap.tables[sel] || []).includes(b.id) ? " mine" : "") +
          (b.id === focus ? " focus" : "") +
          (reused.has(b.id) ? " reused" : "");
        const slots = Array.from({ length: tr.block_size }, (_, s) =>
          '<i class="' + (s < b.used ? "on" : "") + '"></i>'
        ).join("");
        return (
          '<div class="' + cls + '" data-block="' + b.id + '" title="b' + b.id + " · owner " +
          (b.owner || "free") + " · " + b.used + " slots used" + '">' +
          "<b>b" + b.id + '</b><span class="owner">' + (b.owner || "free") + "</span>" +
          '<span class="slots">' + slots + "</span></div>"
        );
      })
      .join("");

    el.innerHTML =
      '<div class="viz-head"><h3>逻辑 token → block table → 物理块池</h3>' +
      '<span class="sub">block_size=' + tr.block_size + " · " + tr.num_blocks + " 块 · 事件 " + esc(snap.label) + "</span></div>" +
      '<div class="reuse-toolbar"><div class="controls">' +
      '<button class="btn ghost" data-nav="-1">←</button>' +
      '<button class="btn" data-play>' + (ctx.playing ? "⏸ 暂停" : "▶ 播放") + "</button>" +
      '<button class="btn ghost" data-nav="1">→</button>' +
      '<input type="range" class="snaprange" data-snap min="0" max="' + (snaps.length - 1) +
      '" value="' + idx + '">' +
      '<span class="tool-note">' + (idx + 1) + " / " + snaps.length + "</span></div></div>" +
      '<div class="compare">' +
      '<div class="stat"><div class="stat-label">池占用</div><div class="stat-value">' +
      used + " / " + tr.num_blocks + '</div><div class="stat-sub">空闲块 ' + snap.free.length + "</div></div>" +
      '<div class="stat"><div class="stat-label">' + esc(sel) + " 长度</div><div class=\"stat-value\">" +
      length + " tok</div><div class=\"stat-sub\">" + table.length + " 块 · 内部浪费 " + waste + " slot</div></div>" +
      '<div class="stat"><div class="stat-label">块表（物理块号）</div><div class="stat-value">' +
      (table.length ? table.map((b) => "b" + b).join(" → ") : "—") +
      '</div><div class="stat-sub">逻辑连续，物理可以不连续</div></div>' +
      "</div>" +
      '<div class="paged-grid">' +
      '<div class="pane"><h4 class="mini-title">逻辑 token（' + esc(sel) + "）</h4>" +
      '<div class="ptoks">' + (tokenChips || '<span class="empty">没有 token</span>') + "</div>" +
      '<p class="note">每个 chip 是一个逻辑位置；按 ' + tr.block_size + " 个一组落进块。</p></div>" +
      '<div class="pane"><h4 class="mini-title">block table（点行高亮物理块）</h4>' +
      '<div class="seqcards">' + seqCards + "</div></div>" +
      '<div class="pane"><h4 class="mini-title">物理块池（8 块）</h4>' +
      '<div class="blockpool" style="grid-template-columns: repeat(4, 1fr)">' + pool + "</div>" +
      '<p class="note">绿色 = 被请求占用；灰 = 空闲；黄 = 本事件新分配；蓝框 = 当前选中序列的块' +
      (reused.size ? "；橙框 = 本次复用的块 b" + [...reused].join("/b") : "") + "。</p></div>" +
      "</div>" +
      '<p class="note">' + esc(step.hint) + "</p>";

    el.querySelectorAll("[data-nav]").forEach((b) =>
      b.addEventListener("click", () => ctx.setSnap(idx + +b.dataset.nav))
    );
    const play = el.querySelector("[data-play]");
    if (play) play.addEventListener("click", () => ctx.togglePlay());
    const range = el.querySelector("[data-snap]");
    if (range) range.addEventListener("change", () => ctx.setSnap(+range.value));
    el.querySelectorAll("[data-seq]").forEach((b) =>
      b.addEventListener("click", () => ctx.setSel(b.dataset.seq))
    );
    el.querySelectorAll("[data-block]").forEach((b) =>
      b.addEventListener("click", () => ctx.setBlock(+b.dataset.block))
    );
  };
})();
