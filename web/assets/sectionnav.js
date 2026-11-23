/* Section navigation for long workshop pages (model anatomy / calculator).
 * Auto-collects h2.sec-title (+ optional [data-toc-sub] groups) from the
 * rendered content, keeps the rail in sync while the content scrolls, and
 * rebuilds when a page re-renders itself (preset / slider changes). */
(function () {
  const LK = (window.LK = window.LK || {});
  const S = (LK.sectionnav = LK.sectionnav || {});
  const esc = (s) =>
    String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function parts(el, fallback) {
    const raw = (el.textContent || "").replace(/\s+/g, " ").trim();
    const m = raw.match(/^(\d+(?:\.\d+)?)\s*[·.、]?\s*(.*)$/);
    if (m && m[2]) return { num: m[1], label: m[2] };
    return { num: String(fallback).padStart(2, "0"), label: raw };
  }

  function collect(content) {
    const nodes = Array.from(content.querySelectorAll("h2.sec-title, [data-toc-sub]"));
    const groups = [];
    let current = null;
    nodes.forEach((el, i) => {
      const top = el.matches("h2.sec-title");
      if (!el.id) el.id = (content.id || "sec") + "-sec-" + (i + 1);
      const p = parts(el, groups.length + 1);
      if (top) {
        current = { el, id: el.id, num: p.num, label: p.label, subs: [] };
        groups.push(current);
      } else if (current) {
        current.subs.push({ el, id: el.id, num: p.num, label: p.label });
      }
    });
    return groups;
  }

  function mount(nav, content) {
    if (!nav || !content) return null;
    const view = nav.closest(".view") || document.scrollingElement || document.documentElement;
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let groups = [];
    let flat = [];
    let activeId = "";
    let frame = 0;
    let lock = 0;

    function render() {
      nav.hidden = groups.length === 0;
      if (!groups.length) {
        nav.innerHTML = "";
        return;
      }
      nav.innerHTML =
        '<div class="tool-toc-head"><span>本节导航</span><b>' + (groups.length ? "1 / " + groups.length : "") + "</b></div>" +
        '<div class="tool-toc-list">' +
        groups
          .map(
            (g) =>
              '<div class="tool-toc-group">' +
              '<button class="toc-link" type="button" data-target="' + g.id + '" title="' + esc(g.num + " " + g.label) + '">' +
              '<span class="toc-no">' + esc(g.num) + "</span>" +
              '<span class="toc-tt">' + esc(g.label) + "</span></button>" +
              (g.subs.length
                ? '<div class="toc-subs">' +
                  g.subs
                    .map(
                      (s) =>
                        '<button class="toc-sub" type="button" data-target="' + s.id + '" title="' +
                        esc(s.num + " " + s.label) + '">' + esc(s.label) + "</button>"
                    )
                    .join("") +
                  "</div>"
                : "") +
              "</div>"
          )
          .join("") +
        "</div>";
    }

    function flatten() {
      flat = [];
      groups.forEach((g, gi) => {
        flat.push({ id: g.id, el: g.el, gi, si: -1 });
        g.subs.forEach((s, si) => flat.push({ id: s.id, el: s.el, gi, si }));
      });
    }

    function topOf() {
      const rect = view.getBoundingClientRect ? view.getBoundingClientRect() : { top: 0 };
      return rect.top + (window.innerWidth <= 980 ? 78 : 96);
    }

    function mark(hit) {
      activeId = hit.id;
      nav.querySelectorAll("[data-target]").forEach((b) => {
        const on = b.dataset.target === hit.id;
        b.classList.toggle("on", on);
        if (on) b.setAttribute("aria-current", "true");
        else b.removeAttribute("aria-current");
      });
      nav.querySelectorAll(".tool-toc-group").forEach((g, gi) => {
        g.classList.toggle("on", gi === hit.gi);
      });
      const pos = nav.querySelector(".tool-toc-head > b");
      if (pos) pos.textContent = hit.gi + 1 + " / " + groups.length;

      // keep the active chip visible in the horizontal (mobile) rail
      const list = nav.querySelector(".tool-toc-list");
      if (list && list.scrollWidth > list.clientWidth + 2) {
        const link = nav.querySelector('[data-target="' + hit.id + '"]');
        if (link) {
          const lr = link.getBoundingClientRect();
          const rr = list.getBoundingClientRect();
          if (lr.left < rr.left - 4 || lr.right > rr.right + 4) {
            list.scrollLeft += lr.left - rr.left - (rr.width - lr.width) / 2;
          }
        }
      }
    }

    function update() {
      if (!flat.length || Date.now() < lock) return;
      const limit = topOf();
      let idx = 0;
      for (let i = 0; i < flat.length; i++) {
        if (flat[i].el.getBoundingClientRect().top <= limit) idx = i;
      }
      const el = view;
      if (el.scrollTop + el.clientHeight >= el.scrollHeight - 4) idx = flat.length - 1;
      mark(flat[idx]);
    }

    function schedule() {
      if (frame) return;
      frame = 1;
      // microtask, not rAF: headless/background tabs pause rAF and the rail
      // would go stale after a page re-render.
      queueMicrotask(() => {
        frame = 0;
        rebuild();
      });
    }

    function rebuild() {
      const keep = activeId;
      groups = collect(content);
      flatten();
      render();
      const hit = keep && flat.find((x) => x.id === keep);
      if (hit) mark(hit);
      else update();
    }

    nav.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-target]");
      if (!btn) return;
      const target = content.querySelector("#" + CSS.escape(btn.dataset.target));
      if (!target) return;
      const hit = flat.find((x) => x.id === btn.dataset.target);
      if (hit) mark(hit);
      lock = Date.now() + 1200; // let the clicked item stay active while programmatic scroll settles
      target.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    });

    ["wheel", "touchmove", "keydown"].forEach((ev) =>
      view.addEventListener(ev, () => { lock = 0; }, { passive: true })
    );
    view.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", () => {
      render();
      update();
    });

    const mo = new MutationObserver(schedule);
    mo.observe(content, { childList: true });

    rebuild();
    return { refresh: rebuild, destroy: () => mo.disconnect() };
  }

  S.mount = mount;
})();
