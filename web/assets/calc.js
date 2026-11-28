/* P4: VRAM / prefill / decode calculator for the anatomy presets. */
(function () {
  const LK = (window.LK = window.LK || {});
  const C = (LK.calc = LK.calc || {});
  const esc = (s) =>
    String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const fmt = (n) => Number(n).toLocaleString("en-US");

  const QUANTS = {
    q4: { label: "Q4_K_M（4.85 bit）", bits: 4.85 },
    q5: { label: "Q5_K_M（5.5 bit）", bits: 5.5 },
    q8: { label: "Q8_0（8.5 bit）", bits: 8.5 },
    fp8: { label: "FP8（8 bit）", bits: 8 },
    bf16: { label: "BF16（16 bit）", bits: 16 },
  };
  const KVDTYPES = {
    fp16: { label: "fp16（2 B）", scale: 1 },
    q8: { label: "q8_0（1 B）", scale: 0.5 },
    q4: { label: "q4_0（0.5 B）", scale: 0.25 },
  };

  const state = {
    presetId: "qwen30b",
    quant: "q4",
    kv: "fp16",
    ctx: 32768,
    batch: 1,
    gpu: "rtx4090",
  };

  // 内部一律用 byte；展示时容量用十进制 GB（与 GPU 的 24GB/80GB 标称一致），
  // 不再把 GB 数值贴上 GiB 标签。KiB/MiB 只用于二进制缓存/块大小的说明。
  const GB = 1e9;
  const GIB = 1073741824;
  function fmtMem(bytes) {
    return bytes >= GB ? (bytes / GB).toFixed(2) + " GB" : (bytes / 1e6).toFixed(0) + " MB";
  }

  C.render = function (el, data) {
    const preset = data.presets.find((p) => p.id === state.presetId) || data.presets[0];
    const m = preset.math;
    const gpu = data.context.gpus.find((g) => g.id === state.gpu) || data.context.gpus[0];
    const q = QUANTS[state.quant];
    const kv = KVDTYPES[state.kv];

    const weightsBytes = (m.total * q.bits) / 8;
    const kvBytesPerToken = m.kv_bytes * kv.scale;
    const kvBytes = state.ctx * state.batch * kvBytesPerToken;
    // 运行时杂项按 1.5 GiB 量级估；明确换算成 byte，避免和十进制 GB 混加。
    const runtimeBytes = 1.5 * GIB;
    const totalBytes = weightsBytes + kvBytes + runtimeBytes;
    const usableBytes = (gpu.unified ? gpu.vram * 0.67 : gpu.vram) * GB;
    const tone = totalBytes > usableBytes ? "bad" : totalBytes > usableBytes - 2 * GB ? "warn" : "ok";

    const prefillFlops =
      2 * m.active * state.ctx +
      4 * preset.layers * preset.q_heads * preset.head_dim * state.ctx * state.ctx;
    const prefillSec = gpu.bf16 ? prefillFlops / (gpu.bf16 * 1e12) : 0;
    const decodeBytes = (m.active * q.bits) / 8 + state.ctx * state.batch * kvBytesPerToken;
    const decodeTok = gpu.bandwidth ? (gpu.bandwidth * 1e9) / decodeBytes : 0;

    const bar = (label, bytes, cls) =>
      '<div class="memrow ' + cls + '"><span>' + label + "</span>" +
      '<div class="membar"><i style="width:' + Math.max(1, (bytes / totalBytes) * 100) + '%"></i></div>' +
      "<b>" + fmtMem(bytes) + "</b></div>";

    el.innerHTML =
      '<p class="kicker">P4 · 显存与速度估算</p>' +
      '<h1 class="page-title">这套 LLM 跑在这张卡上，要多少显存？</h1>' +
      '<p class="page-lead">参数、量化、上下文、并发、KV 精度——五个滑杆决定同一组数字。' +
      "公式和「本次学习设定」一致：权重看参数量，KV 看上下文 × 层数 × KV 头，速度看带宽和算力。</p>" +
      '<h2 class="sec-title">1 · 输入：LLM / 精度 / 上下文 / 并发 / GPU</h2>' +
      '<div class="calc-layout"><div class="calc-controls">' +
      '<label class="calc-field"><span>LLM</span><select data-field="preset">' +
      data.presets
        .map(
          (p) =>
            '<option value="' + p.id + '"' + (p.id === preset.id ? " selected" : "") + ">" +
            esc(p.name) + "</option>"
        )
        .join("") +
      "</select></label>" +
      '<label class="calc-field"><span>权重精度</span><select data-field="quant">' +
      Object.entries(QUANTS)
        .map(
          ([id, v]) =>
            '<option value="' + id + '"' + (id === state.quant ? " selected" : "") + ">" + v.label + "</option>"
        )
        .join("") +
      "</select></label>" +
      '<label class="calc-field"><span>KV 精度</span><select data-field="kv">' +
      Object.entries(KVDTYPES)
        .map(
          ([id, v]) =>
            '<option value="' + id + '"' + (id === state.kv ? " selected" : "") + ">" + v.label + "</option>"
        )
        .join("") +
      "</select></label>" +
      '<label class="calc-field"><span>上下文 <b>' + fmt(state.ctx) + " tok</b></span>" +
      '<input type="range" data-field="ctx" min="2048" max="262144" step="2048" value="' + state.ctx + '"></label>' +
      '<label class="calc-field"><span>并发请求 <b>' + state.batch + "</b></span>" +
      '<input type="range" data-field="batch" min="1" max="8" step="1" value="' + state.batch + '"></label>' +
      '<div class="calc-field"><span>GPU</span><div class="gpu-row">' +
      data.context.gpus
        .map(
          (g) =>
            '<button class="gpu' + (g.id === gpu.id ? " on" : "") + '" data-gpu="' + g.id + '">' +
            esc(g.name) + "</button>"
        )
        .join("") +
      "</div></div>" +
      '<div class="gpu-specs">' +
      '<div class="spec"><span>显存</span><b>' + gpu.vram + " GB" + (gpu.unified ? "（统一）" : "") + "</b></div>" +
      '<div class="spec"><span>带宽</span><b>' + gpu.bandwidth + " GB/s</b></div>" +
      '<div class="spec"><span>BF16 算力</span><b>' + gpu.bf16 + " TFLOPS</b></div>" +
      '<div class="spec"><span>decode 理论</span><b>' + Math.round(decodeTok) + " tok/s</b></div>" +
      "</div></div>" +
      '<div class="calc-result">' +
      '<h2 class="sec-title">2 · 显存账本：权重 + KV + 运行时</h2>' +
      '<div class="compare">' +
      '<div class="stat"><div class="stat-label">总占用</div><div class="stat-value">' +
      fmtMem(totalBytes) + '</div><div class="stat-sub">权重 ' + fmtMem(weightsBytes) + " + KV " +
      fmtMem(kvBytes) + " + 运行时 " + fmtMem(runtimeBytes) + "</div></div>" +
      '<div class="stat"><div class="stat-label">这条配置能跑吗</div><div class="stat-value ' + tone +
      '">' + (tone === "ok" ? "宽裕" : tone === "warn" ? "紧张" : "放不下") +
      '</div><div class="stat-sub">可用预算 ≈ ' + fmtMem(usableBytes) + "（" +
      (gpu.unified ? "统一内存按 67% 估算" : "显存（十进制 GB）") + "）</div></div>" +
      "</div>" +
      '<div class="membars">' + bar("权重", weightsBytes, "w") + bar("KV cache", kvBytes, "k") +
      bar("运行时", runtimeBytes, "r") + "</div>" +
      '<h2 class="sec-title">3 · 速度上限：prefill 与 decode</h2>' +
      '<div class="compare calc-est">' +
      '<div class="stat"><div class="stat-label">prefill 计算量（n=' + fmt(state.ctx) + '）</div><div class="stat-value">' +
      (prefillFlops / 1e12).toFixed(1) + ' TF</div><div class="stat-sub">n=' +
      fmt(state.ctx) + " 时约 " + (prefillFlops / 1e12).toFixed(0) +
      " TF；含 GEMM + full attention</div></div>" +
      '<div class="stat"><div class="stat-label">prefill 时间下限</div><div class="stat-value">' +
      prefillSec.toFixed(2) + ' s</div><div class="stat-sub">假设打满 ' + gpu.bf16 +
      " TFLOPS；实际差 2–3 倍</div></div>" +
      '<div class="stat"><div class="stat-label">decode 带宽上限</div><div class="stat-value">' +
      Math.round(decodeTok) + ' tok/s</div><div class="stat-sub">每步读激活权重 + 全部 KV</div></div>' +
      "</div>" +
      '<h2 class="sec-title">4 · 公式与口径</h2>' +
      '<p class="note">同一套公式：<code>权重 = 参数量 × 每参数字节</code>，' +
      "<code>KV = 上下文 × 层数 × KV 头 × head_dim × 2 × 字节</code>，" +
      "<code>decode ≈ 带宽 / (激活权重 + KV)</code>。并发请求各自占一份 KV；prefix cache 共享前缀时才会计数复用。</p>" +
      "</div></div>";

    el.querySelectorAll("[data-field]").forEach((inp) => {
      inp.addEventListener("change", () => {
        const field = inp.dataset.field;
        const key = field === "preset" ? "presetId" : field;
        state[key] = inp.tagName === "SELECT" ? inp.value : +inp.value;
        C.render(el, data);
      });
    });
    el.querySelectorAll("[data-gpu]").forEach((b) =>
      b.addEventListener("click", () => {
        state.gpu = b.dataset.gpu;
        C.render(el, data);
      })
    );
  };
})();
