/* Minimal Python syntax highlighter + line renderer.
 * No dependencies; whole-file scan so triple-quoted docstrings work. */
(function () {
  const LK = (window.LK = window.LK || {});

  const KEYWORDS = new Set(
    ("False None True and as assert async await break class continue def del elif else " +
      "except finally for from global if import in is lambda nonlocal not or pass raise " +
      "return try while with yield").split(" ")
  );
  const BUILTINS = new Set(
    ("print len range int float bool str list tuple dict set type super isinstance " +
      "enumerate zip max min sum abs open sorted reversed repr format getattr setattr " +
      "hasattr iter next map filter any all round divmod hash id").split(" ")
  );

  const TOKEN_RE =
    /("""[\s\S]*?"""|'''[\s\S]*?''')|(#[^\n]*)|("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(@[A-Za-z_]\w*)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)|(\n)|([\s\S])/g;

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function span(cls, text) {
    return '<span class="' + cls + '">' + esc(text) + "</span>";
  }

  function highlight(text) {
    const lines = [];
    let cur = "";
    TOKEN_RE.lastIndex = 0;
    let m;

    // Tokens (triple-quoted strings) can span lines: split them so every
    // rendered line still maps 1:1 onto a source line.
    function emit(cls, tok) {
      const parts = tok.split("\n");
      for (let k = 0; k < parts.length; k++) {
        if (parts[k]) cur += span(cls, parts[k]);
        if (k < parts.length - 1) { lines.push(cur); cur = ""; }
      }
    }

    while ((m = TOKEN_RE.exec(text)) !== null) {
      const [tok, triple, comment, string, decorator, number, ident, newline, other] = m;
      if (newline) {
        lines.push(cur);
        cur = "";
        continue;
      }
      if (triple) { emit("s", tok); continue; }
      if (comment) { emit("c", tok); continue; }
      if (string) { emit("s", tok); continue; }
      if (decorator) { cur += span("d", tok); continue; }
      if (number) { cur += span("n", tok); continue; }
      if (ident) {
        if (KEYWORDS.has(ident)) cur += span("k", ident);
        else if (BUILTINS.has(ident)) cur += span("b", ident);
        else {
          // A name immediately followed by "(" is a call: colour it like a function.
          let j = TOKEN_RE.lastIndex;
          while (j < text.length && (text[j] === " " || text[j] === "\t")) j++;
          cur += span(text[j] === "(" ? "f" : "o", ident);
        }
        continue;
      }
      cur += esc(other);
    }
    lines.push(cur);
    return lines;
  }

  LK.codeview = {
    mount(container, sourceLines) {
      const html = highlight(sourceLines.join("\n"));
      const frag = document.createDocumentFragment();
      const els = [];
      for (let i = 0; i < html.length; i++) {
        const row = document.createElement("div");
        row.className = "ln";
        row.dataset.line = String(i + 1);
        const no = document.createElement("span");
        no.className = "no";
        no.textContent = String(i + 1);
        const src = document.createElement("span");
        src.className = "src";
        src.innerHTML = html[i] || " ";
        row.appendChild(no);
        row.appendChild(src);
        frag.appendChild(row);
        els.push(row);
      }
      container.innerHTML = "";
      container.appendChild(frag);

      function clear() {
        for (const el of els) el.classList.remove("hl", "hl-b");
      }

      function highlightRanges(ranges) {
        clear();
        if (!ranges || !ranges.length) return;
        for (const r of ranges) {
          for (let n = r.start; n <= r.end && n <= els.length; n++) {
            const el = els[n - 1];
            if (!el) continue;
            el.classList.add("hl");
            if (r.tone === "b") el.classList.add("hl-b");
          }
        }
      }

      // Only keep the line ranges that belong to the current step. Line
      // numbers stay original, so the snippet is still traceable to the file.
      function showOnly(ranges) {
        const keep = new Set();
        for (const r of ranges || []) {
          for (let n = r.start; n <= r.end && n <= els.length; n++) keep.add(n);
        }
        for (let i = 0; i < els.length; i++) {
          const visible = keep.has(i + 1);
          els[i].hidden = !visible;
          els[i].style.display = visible ? "" : "none";
        }
      }

      function scrollToLine(line) {
        const el = els[line - 1];
        if (!el) return;
        const cRect = container.getBoundingClientRect();
        const eRect = el.getBoundingClientRect();
        if (eRect.top >= cRect.top + 40 && eRect.bottom <= cRect.bottom - 40) return;
        // Content offset of the line, independent of offsetParent quirks.
        const top =
          eRect.top - cRect.top + container.scrollTop;
        container.scrollTop = Math.max(0, top - container.clientHeight * 0.32);
      }

      return { highlight: highlightRanges, showOnly, scrollToLine };
    },
  };
})();
