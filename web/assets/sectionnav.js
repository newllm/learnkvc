/* Section navigation for long workshop pages (model anatomy / calculator).
 * Same rail language as the chapter navigation: numbered items, collapse
 * toggle, scroll-spy and localStorage state. Content is auto-collected from
 * h2.sec-title (+ optional [data-toc-sub] groups) and rebuilt whenever the
 * page re-renders itself (preset / slider changes). */
(function () {
  const LK = (window.LK = window.LK || {});
  const S = (LK.sectionnav = LK.sectionnav || {});
  const STORE = "whykvc.sectionnav";
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
      if (!el.id) el.id = (content.id || "tool") + "-sec-" + (i + 1);
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

  function readStore() {
    try {
      return JSON.parse(localStorage.getItem(STORE) || "{}") || {};
    } catch (e) {
      return {};
    }
  }

  function saveStore(key, collapsed) {
    try {
      const all = readStore();
      all[key] = collapsed;
      localStorage.setItem(STORE, JSON.stringify(all));
    } catch (e) {
      /* localStorage unavailable */
    }
  }

  function mount(nav, content) {
    if (!nav || !content) return null;
    const layout = nav.closest(".tool-layout");
    const viewEl = nav.closest(".view");
    const view = viewEl || document.scrollingElement || document.documentElement;
    const key = content.id || "tool";
    const reduce =
      window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const saved = readStore();
    let collapsed =
      typeof saved[key] === "boolean" ? saved[key] : window.innerWidth <= 980;
    let groups = [];
    let flat = [];
    let activeId = "";
    let frame = 0;
    let lock = 0;

    function syncTopbar() {
      const bar = document.querySelector(".topbar");
      const h = bar ? Math.round(bar.getBoundingClientRect().height) : 54;
      document.documentElement.style.setProperty("--topbar-h", h + "px");
    }

    function syncNavOffset() {
      if (!viewEl) return;
      const h = nav.hidden ? 0 : Math.round(nav.getBoundingClientRect().height);
      viewEl.style.setProperty("--tool-nav-h", h + "px");
    }

    function applyCollapsed() {
      if (layout) layout.classList.toggle("nav-collapsed", collapsed);
      const btn = nav.querySelector(".stepnav-toggle");
      if (!btn) return;
      btn.textContent = collapsed ? "›" : "‹";
      btn.setAttribute("aria-expanded", String(!collapsed));
      btn.setAttribute("aria-label", collapsed ? "展开小节导航" : "收起小节导航");
    }

    function render() {
      nav.hidden = groups.length === 0;
      if (!groups.length) {
        nav.innerHTML = "";
        return;
      }
      nav.innerHTML =
        '<div class="stepnav-head"><span>小节导航</span><b>1 / ' + groups.length + "</b>" +
        '<button class="stepnav-toggle" type="button" aria-expanded="true" aria-label="收起小节导航">‹</button></div>' +
        '<div class="tool-toc-body">' +
        groups
          .map(
            (g, gi) =>
              '<button class="stage-nav-item tool-sec" type="button" data-target="' + g.id + '" title="' +
              esc(g.num + " " + g.label) + '">' +
              '<span class="sid">' + String(gi + 1).padStart(2, "0") + "</span>" +
              '<span class="stt">' + esc(g.label) + "</span>" +
              "<em>" + (g.subs.length ? g.subs.length + " 小节" : "") + "</em></button>" +
              (g.subs.length
                ? '<div class="tool-subs">' +
                  g.subs
                    .map(
                      (s) =>
                        '<button class="stepnav-item tool-sub" type="button" data-target="' + s.id + '" title="' +
                        esc(s.num + " " + s.label) + '">' +
                        '<span class="no">' + esc(s.num) + "</span>" +
                        '<span class="tt">' + esc(s.label) + "</span></button>"
                    )
                    .join("") +
                  "</div>"
                : "")
          )
          .join("") +
        "</div>";
      applyCollapsed();
      syncNavOffset();
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
        const top = b.classList.contains("tool-sec");
        const on = b.dataset.target === hit.id;
        b.classList.toggle(top ? "on" : "active", on);
        if (on) b.setAttribute("aria-current", "true");
        else b.removeAttribute("aria-current");
      });
      // keep the parent section highlighted while one of its subsections is active
      nav.querySelectorAll(".tool-sec").forEach((b, gi) => b.classList.toggle("on", gi === hit.gi));
      const pos = nav.querySelector(".stepnav-head > b");
      if (pos) pos.textContent = hit.gi + 1 + " / " + groups.length;

      const body = nav.querySelector(".tool-toc-body");
      const link = nav.querySelector('[data-target="' + hit.id + '"]');
      if (body && link && body.scrollHeight > body.clientHeight + 4) {
        const lr = link.getBoundingClientRect();
        const br = body.getBoundingClientRect();
        if (lr.top < br.top || lr.bottom > br.bottom) {
          body.scrollTop += lr.top - br.top - 8;
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
      if (e.target.closest(".stepnav-toggle")) {
        collapsed = !collapsed;
        saveStore(key, collapsed);
        applyCollapsed();
        syncNavOffset();
        return;
      }
      const btn = e.target.closest("[data-target]");
      if (!btn) return;
      const target = content.querySelector("#" + CSS.escape(btn.dataset.target));
      if (!target) return;
      const hit = flat.find((x) => x.id === btn.dataset.target);
      if (hit) mark(hit);
      lock = Date.now() + 1200; // keep the clicked item active while the scroll settles
      target.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    });

    ["wheel", "touchmove", "keydown"].forEach((ev) =>
      view.addEventListener(ev, () => { lock = 0; }, { passive: true })
    );
    view.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", () => {
      syncTopbar();
      applyCollapsed();
      syncNavOffset();
      update();
    });

    const mo = new MutationObserver(schedule);
    mo.observe(content, { childList: true });

    syncTopbar();
    rebuild();
    syncNavOffset();
    return { refresh: rebuild, destroy: () => mo.disconnect() };
  }

  S.mount = mount;
})();
