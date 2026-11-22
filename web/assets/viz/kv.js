/* Chapter 1 visualizations. Each renderer returns a DOM tree rebuilt from
 * the trace + current genStep; controls call back into the app. */
(function () {
  const LK = (window.LK = window.LK || {});
  const viz = (LK.viz = LK.viz || {});

  const fmt = (n) => Number(n).toLocaleString("en-US");

  function controls(g, max, ctx) {
    const playing = !!ctx.playing;
    return (
      '<div class="gen-controls">' +
      '<button class="btn ghost" data-act="reset" title="回到 0">⏮</button>' +
      '<button class="btn ' + (playing ? "on" : "") + '" data-act="play">' +
      (playing ? "⏸ 暂停" : "▶ 播放") +
      "</button>" +
      '<input type="range" min="0" max="' + max + '" value="' + g + '" data-act="range">' +
      "<span>t = " + g + " / " + max + "</span>" +
      "</div>"
    );
  }

  function bindControls(root, ctx) {
    root.querySelectorAll("[data-act]").forEach((el) => {
      const act = el.dataset.act;
      if (act === "play") el.addEventListener("click", () => ctx.togglePlay());
      if (act === "reset") el.addEventListener("click", () => ctx.setGenStep(0));
      if (act === "range") el.addEventListener("change", () => ctx.setGenStep(+el.value));
    });
  }

  function stepTable(tr, g, max) {
    const rows = tr.steps
      .map((s) => {
        const cls = s.s < g ? "done" : s.s === g ? "next" : "pending";
        return (
          '<tr class="' + cls + '">' +
          "<td>" + s.s + "</td>" +
          "<td>" + s.token + "</td>" +
          "<td>" + s.plain_input + "</td>" +
          "<td>" + s.cached_input + "</td>" +
          "<td>" + fmt(s.plain_pos) + " / " + fmt(s.cached_pos) + "</td>" +
          "<td>" + fmt(s.plain_pairs) + " / " + fmt(s.cached_pairs) + "</td>" +
          "</tr>"
        );
      })
      .join("");
    return (
      '<div class="table-wrap"><table class="dualtable">' +
      "<thead><tr><th>步骤</th><th>新 token</th><th>无 cache 输入</th>" +
      "<th>有 cache 输入</th><th>位置 无 / 有</th><th>对数 无 / 有</th></tr></thead>" +
      "<tbody>" + rows + "</tbody></table></div>"
    );
  }

  /* ------------------------------------------------------------- dual */

  function dual(el, step, ctx) {
    const tr = ctx.trace;
    const g = ctx.genStep;
    const max = tr.generated.length;
    const last = g ? tr.steps[g - 1] : null;
    const next = g < max ? tr.steps[g] : null;

    const promptChips = tr.prompt
      .map((t, i) => '<span class="chip prompt"><span class="idx">P' + (i + 1) + "</span>" + t + "</span>")
      .join("");
    const genChips = tr.generated
      .slice(0, g)
      .map((t, i) => '<span class="chip gen"><span class="idx">G' + (i + 1) + "</span>" + t + "</span>")
      .join("");
    const pending = g < max ? '<span class="chip pending">?</span>' : "";

    const plainInput = next ? next.plain_input : "—";
    const cachedInput = next ? next.cached_input : "—";
    const plainBig = next ? "下一步输入 " + plainInput + " 个 token" : "已完成全部 " + max + " 个 step";
    const cachedBig = next ? "下一步输入 " + cachedInput + " 个 token" : "已完成全部 " + max + " 个 step";
    const plainPos = last ? last.plain_pos : 0;
    const cachedPos = last ? last.cached_pos : 0;
    const plainPairs = last ? last.plain_pairs : 0;
    const cachedPairs = last ? last.cached_pairs : 0;
    const cacheLen = last ? last.cache_len : 0;

    el.innerHTML =
      '<div class="viz-head"><h3>两条路径并排 · 生成进度可播放</h3>' + controls(g, max, ctx) + "</div>" +
      '<div class="chips">' + promptChips + genChips + pending + "</div>" +
      '<div class="lanes">' +
      '<div class="lane"><h4>无 cache（每步重喂整段前缀）</h4>' +
      '<div class="big">' + plainBig + "</div>" +
      '<div class="small">累计位置前向 ' + fmt(plainPos) + " · attention 对数 " + fmt(plainPairs) + "</div></div>" +
      '<div class="lane"><h4>有 cache（prefill 一次，之后每步 1 个）</h4>' +
      '<div class="big">' + cachedBig + "</div>" +
      '<div class="small">缓存长度 ' + fmt(cacheLen) + " · 累计位置前向 " + fmt(cachedPos) +
      " · attention 对数 " + fmt(cachedPairs) + "</div></div>" +
      "</div>" +
      stepTable(tr, g, max) +
      '<p class="chip-label">prompt 8 个 token，greedy 生成 8 个：t = 0 时两条路径都从 prefill 开始；' +
      "t &gt; 0 后无 cache 每步重算 8…15 个位置，有 cache 每步只算 1 个。表格里数值是累计量。</p>";
    bindControls(el, ctx);
  }

  /* ------------------------------------------------------------ cache */

  function cache(el, step, ctx) {
    const tr = ctx.trace;
    const g = ctx.genStep;
    const max = tr.generated.length;
    const cacheLen = g ? tr.steps[g - 1].cache_len : 0;
    const cols = tr.prompt.length + max; // 16 positions in total

    let header = "<span></span>";
    for (let i = 0; i < cols; i++) {
      const label = i < tr.prompt.length ? "P" + (i + 1) : "G" + (i - tr.prompt.length + 1);
      header += '<span class="colhead">' + label + "</span>";
    }
    let rows = "";
    for (let layer = 0; layer < tr.model.layers; layer++) {
      rows += '<span class="rowlabel">L' + layer + "</span>";
      for (let i = 0; i < cols; i++) {
        let cls = "cell";
        if (i < cacheLen) cls += " filled";
        if (i === cacheLen - 1 && g > 0) cls += " new";
        rows += '<span class="' + cls + '"></span>';
      }
    }

    el.innerHTML =
      '<div class="viz-head"><h3>每层的 KV 缓存</h3>' + controls(g, max, ctx) + "</div>" +
      '<div class="grid-note">LayerKVCache.key: (1, ' + tr.model.kv_heads + ", T, " + tr.model.head_dim + ") · T = " +
      cacheLen + "（2 层的长度同步推进）</div>" +
      '<div class="cachegrid" style="grid-template-columns: 46px repeat(' + cols + ', 1fr)">' +
      header + rows + "</div>" +
      '<p class="chip-label">P 是 prompt 的 8 个位置，G 是生成的 8 个位置；黄色是本次 step 新写入的 K/V。' +
      "有 cache 时 t &gt; 0 每步只新增 1 列，但 attention 仍能读到之前所有列。</p>";
    bindControls(el, ctx);
  }

  /* ------------------------------------------------------ experiments */

  function experiments(el, step, ctx) {
    const f = ctx.trace.features;
    el.innerHTML =
      '<div class="viz-head"><h3>缓存的三个额外操作</h3><span class="sub">来自 check_cache_features()</span></div>' +
      '<div class="expgrid">' +
      '<div class="exp" id="exp-window"><h4>滑动窗口</h4>' +
      '<div class="row">' +
      Array.from({ length: f.window.feed }, (_, i) => '<span class="mini" data-i="' + i + '">' + i + "</span>").join("") +
      "</div>" +
      '<button class="btn" data-exp="window">max_length=' + f.window.max_length + " 喂 " + f.window.feed + " 个 token</button>" +
      '<div class="out"></div></div>' +
      '<div class="exp" id="exp-reorder"><h4>beam reorder</h4>' +
      '<div class="row" data-row="0">' + [0, 1, 2].map((i) => '<span class="mini" data-v="' + i + '">' + i + "</span>").join("") + "</div>" +
      '<div class="row" data-row="1">' + [3, 4, 5].map((i) => '<span class="mini" data-v="' + i + '">' + i + "</span>").join("") + "</div>" +
      '<button class="btn" data-exp="reorder">reorder([' + f.reorder.indices.join(", ") + "])</button>" +
      '<div class="out"></div></div>' +
      '<div class="exp" id="exp-fork"><h4>fork 分支</h4>' +
      '<div class="row" data-parent><span class="mini">0</span><span class="mini">1</span></div>' +
      '<div class="row" data-branch style="display:none"><span class="mini">0</span><span class="mini">1</span><span class="mini new">2</span></div>' +
      '<button class="btn" data-exp="fork">fork() 再 append 1 个</button>' +
      '<div class="out"></div></div>' +
      "</div>";

    const out = (sel, text) => (el.querySelector(sel + " .out").textContent = text);

    el.querySelector('[data-exp="window"]').addEventListener("click", () => {
      const kept = new Set(f.window.kept_positions);
      el.querySelectorAll("#exp-window .mini").forEach((m) => {
        const i = +m.dataset.i;
        m.classList.add(kept.has(i) ? "keep" : "drop");
      });
      out("#exp-window", "length = " + f.window.length + "，丢弃 " + f.window.dropped + " 个最早的位置");
    });

    el.querySelector('[data-exp="reorder"]').addEventListener("click", () => {
      const rows = [...el.querySelectorAll("#exp-reorder .row")];
      const a = rows[0].innerHTML, b = rows[1].innerHTML;
      rows[0].innerHTML = b; rows[1].innerHTML = a;
      out("#exp-reorder", "reorder([1, 0])：两层缓存按同一组下标交换 beam 行");
    });

    el.querySelector('[data-exp="fork"]').addEventListener("click", () => {
      el.querySelector("#exp-fork [data-branch]").style.display = "flex";
      out("#exp-fork", "parent.length = " + f.fork.parent_len + "，branch.length = " + f.fork.branch_len + "（clone 后再 append）");
    });
  }

  /* ------------------------------------------------------------- mask */

  function mask(el) {
    const qAbs = [2, 3, 4];
    const kAbs = [1, 2, 3, 4];
    function grid(rows) {
      let html = '<div class="maskgrid" style="grid-template-columns: 34px repeat(' + kAbs.length + ', 1fr)"><span></span>';
      for (const k of kAbs) html += '<span class="colhead">K' + k + "</span>";
      rows.forEach((row, qi) => {
        html += '<span class="rowlabel">Q' + qAbs[qi] + "</span>";
        row.forEach((cell) => {
          html += '<span class="m ' + (cell.ok ? "ok" : "") + (cell.diff ? " diff" : "") + '"></span>';
        });
      });
      html += "</div>";
      return html;
    }
    const correct = [], naive = [];
    qAbs.forEach((q, qi) => {
      const crow = [], nrow = [];
      kAbs.forEach((k, ki) => {
        const ok = k <= q; // absolute positions
        const nOk = ki <= qi; // naive index mask (assumes key 0 is position 0)
        crow.push({ ok, diff: ok !== nOk });
        nrow.push({ ok: nOk, diff: ok !== nOk });
      });
      correct.push(crow);
      naive.push(nrow);
    });

    el.innerHTML =
      '<div class="viz-head"><h3>掩码：绝对位置 vs 朴素下三角</h3><span class="sub">窗口丢掉 2 个 key，past = 2，q_len = 3</span></div>' +
      '<div class="maskwrap">' +
      '<div><div class="grid-note">错误：直接用 is_causal（按 key 下标）</div>' + grid(naive) + "</div>" +
      '<div><div class="grid-note">正确：first_key_position + 绝对位置</div>' + grid(correct) + "</div>" +
      "</div>" +
      '<div class="mask-legend">绿 = 允许看；红框 = 两种掩码不一致的位置（朴素掩码把已经丢掉的绝对位置当成了 0）</div>';
  }

  /* ----------------------------------------------------------- shapes */

  function shapes(el, step, ctx) {
    const m = ctx.trace.model;
    const proj = [
      ["Wq", m.d_model + " × " + m.q_heads * m.head_dim],
      ["Wk", m.d_model + " × " + m.kv_heads * m.head_dim],
      ["Wv", m.d_model + " × " + m.kv_heads * m.head_dim],
      ["Wo", m.d_model + " × " + m.d_model],
      ["W1", m.d_model + " × " + 2 * m.d_model],
      ["W2", 2 * m.d_model + " × " + m.d_model],
    ];
    el.innerHTML =
      '<div class="viz-head"><h3>TinyModel 的形状</h3><span class="sub">2 层，4/2 头 GQA</span></div>' +
      '<div class="config">' +
      "<span class=\"cfg\">layers <b>" + m.layers + "</b></span>" +
      "<span class=\"cfg\">Q / KV heads <b>" + m.q_heads + " / " + m.kv_heads + "</b></span>" +
      "<span class=\"cfg\">head_dim <b>" + m.head_dim + "</b></span>" +
      "<span class=\"cfg\">d_model <b>" + m.d_model + "</b></span>" +
      "<span class=\"cfg\">vocab <b>" + m.vocab + "</b></span>" +
      "</div>" +
      '<table class="shapetable"><thead><tr><th>权重</th><th>形状</th></tr></thead><tbody>' +
      proj.map((p) => "<tr><td>" + p[0] + "</td><td>" + p[1] + "</td></tr>").join("") +
      "</tbody></table>" +
      '<p class="chip-label">每层缓存一组 K/V，布局 (1, ' + m.kv_heads + ", T, " + m.head_dim +
      ")；forward 只返回最后位置的 logits，generate 的 use_cache 只改变每步喂多少 token。</p>";
  }

  /* --------------------------------------------------------- numerics */

  function numerics(el, step, ctx) {
    const tr = ctx.trace;
    const diff = tr.numerics.diff;
    const threshold = tr.numerics.threshold;
    const pct = Math.max(1, Math.min(100, (diff / threshold) * 100));
    el.innerHTML =
      '<div class="viz-head"><h3>两条路径的数值完全一致</h3><span class="sub">check_numerics()</span></div>' +
      '<div class="num"><span class="value">' + Number(diff).toExponential(2) + "</span>" +
      '<span class="unit">max |plain − cached| &lt; ' + threshold.toExponential(0) + "</span></div>" +
      '<div class="num-bar"><i style="width:' + pct + '%"></i></div>' +
      '<p class="note">同一段 <code>[prefix + token]</code>：一次性 forward 与 prefill + decode 的 logits 最大差异 ' +
      Number(diff).toExponential(2) + "（浮点累加顺序），argmax 与生成 token 完全一致。" +
      (tr.source === "analytic" ? " 当前为 README 记录的数值；安装 torch 后重新 build 会换成本机实测。" : "") +
      "</p>";
  }

  /* ------------------------------------------------------------ entry */

  viz.render = function (el, step, ctx) {
    switch (step.viz) {
      case "dual": return dual(el, step, ctx);
      case "cache": return cache(el, step, ctx);
      case "experiments": return experiments(el, step, ctx);
      case "mask": return mask(el, step, ctx);
      case "shapes": return shapes(el, step, ctx);
      case "numerics": return numerics(el, step, ctx);
      default: el.innerHTML = "";
    }
  };
})();
