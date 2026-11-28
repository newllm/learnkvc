#!/usr/bin/env python3
"""Build web/data/data.js for the interactive reader.

Sources of truth:
  example/kv_cache.py          -> code lines + AST symbol ranges
  example/KV_CACHE_READING.md  -> explanation cards (sections -> HTML)
  README.md                    -> overview text
  web/tools/steps.json         -> chapter step list (symbol refs, viz names)
  web/tools/collect_traces.py  -> state snapshots (web/data/traces.js)

Run:  python3 web/tools/build.py
"""

from __future__ import annotations

import ast
import html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "example"
WEB = ROOT / "web"
DATA = WEB / "data"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_traces  # noqa: E402


# --------------------------------------------------------------------------- #
# GitHub-style heading anchors (same rule the repo already links with)
# --------------------------------------------------------------------------- #

def gh_slug(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.lower()
    text = re.sub(r"[^\w\u4e00-\u9fff\- ]", "", text)
    return text.strip().replace(" ", "-")


def parse_sections(path: Path) -> tuple[list[dict], list[str]]:
    """Split a markdown file into ``##`` / ``###`` sections.

    Returns (sections, intro_lines); intro_lines is everything before the
    first heading (the H1 title excluded).
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    sections: list[dict] = []
    intro: list[str] = []
    current: dict | None = None
    in_code = False
    seen_h1 = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code = not in_code
            if current is not None:
                current["body"].append(line)
            continue
        m = None
        if not in_code:
            m = re.match(r"^(#{2,4}) (.+?)\s*$", line)
        if m:
            level = len(m.group(1))
            title = m.group(2)
            current = {"level": level, "title": title, "id": gh_slug(title), "body": []}
            sections.append(current)
            continue
        if not in_code and line.startswith("# ") and not seen_h1:
            seen_h1 = True
            continue
        if current is None:
            intro.append(line)
        else:
            current["body"].append(line)
    # Trim trailing blank lines from bodies.
    for sec in sections:
        while sec["body"] and not sec["body"][-1].strip():
            sec["body"].pop()
        while sec["body"] and not sec["body"][0].strip():
            sec["body"].pop(0)
    return sections, intro


# --------------------------------------------------------------------------- #
# Minimal markdown -> HTML (covers the syntax used by the reading guides)
# --------------------------------------------------------------------------- #

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def inline_md(text: str) -> str:
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", out)

    def link(m: re.Match) -> str:
        label, href = m.group(1), html.unescape(m.group(2))
        if not re.match(r"^(https?:|#|/)", href):
            href = "../example/" + href
        return f'<a href="{html.escape(href)}" target="_blank" rel="noopener">{label}</a>'

    out = LINK_RE.sub(link, out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    return out


def _is_cjk(ch: str) -> bool:
    return bool(ch) and ("\u4e00" <= ch <= "\u9fff")


def join_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not text:
            text = line
            continue
        sep = "" if (_is_cjk(text[-1]) or _is_cjk(line[0])) else " "
        text += sep + line
    return text


def _split_table_row(line: str) -> list[str]:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


def render_md(lines: list[str]) -> str:
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            i += 1
            code: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            cls = f' class="lang-{html.escape(lang)}"' if lang else ""
            out.append(
                f'<pre class="md-code"><code{cls}>'
                + html.escape("\n".join(code))
                + "</code></pre>"
            )
            continue

        if stripped.startswith("|") and i + 1 < n and re.match(
            r"^\s*\|?\s*:?-{2,}", lines[i + 1]
        ):
            head = _split_table_row(line)
            i += 2
            rows: list[list[str]] = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(_split_table_row(lines[i]))
                i += 1
            cells = "".join(f"<th>{inline_md(cell)}</th>" for cell in head)
            body = "".join(
                "<tr>" + "".join(f"<td>{inline_md(cell)}</td>" for cell in row) + "</tr>"
                for row in rows
            )
            out.append(f"<table><thead><tr>{cells}</tr></thead><tbody>{body}</tbody></table>")
            continue

        if stripped.startswith(">"):
            quote: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].lstrip())
                i += 1
            out.append(f"<blockquote>{render_md(quote)}</blockquote>")
            continue

        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            ordered = m.group(2)[0].isdigit()
            items: list[str] = []
            while i < n:
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if not mm:
                    break
                indent = len(mm.group(1))
                items.append(("  " if indent >= 2 else "") + inline_md(mm.group(3)))
                i += 1
            tag = "ol" if ordered else "ul"
            lis = "".join(f'<li class="{"sub" if it.startswith("  ") else ""}">{it.strip()}</li>' for it in items)
            out.append(f"<{tag}>{lis}</{tag}>")
            continue

        if not stripped:
            i += 1
            continue

        para: list[str] = []
        while i < n and lines[i].strip() and not re.match(
            r"^\s*(```|>|\||[-*]\s|\d+\.\s|#{2,4}\s)", lines[i]
        ):
            para.append(lines[i])
            i += 1
        out.append(f"<p>{inline_md(join_lines(para))}</p>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# AST symbol -> line ranges
# --------------------------------------------------------------------------- #

def symbol_ranges(path: Path) -> dict[str, tuple[int, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    ranges: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            ranges[f"function:{node.name}"] = (node.lineno, node.end_lineno)
        elif isinstance(node, ast.ClassDef):
            ranges[f"class:{node.name}"] = (node.lineno, node.end_lineno)
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    ranges[f"class:{node.name}.{sub.name}"] = (sub.lineno, sub.end_lineno)
    if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        ranges["docstring:module"] = (tree.body[0].lineno, tree.body[0].end_lineno)
    return ranges


# --------------------------------------------------------------------------- #
# Model anatomy (params / dimensions / KV) — computed from preset configs
# --------------------------------------------------------------------------- #

def model_math(c: dict) -> dict:
    layers, h = c["layers"], c["hidden"]
    qh, kh, hd = c["q_heads"], c["kv_heads"], c["head_dim"]
    ffn, experts, einter, topk = c["ffn"], c["experts"], c["expert_inter"], c["top_k"]
    vocab = c["vocab"]
    q_dim, kv_dim = qh * hd, kh * hd
    attn = h * q_dim + 2 * (h * kv_dim) + q_dim * h + 2 * hd
    router = h * experts if experts else 0
    ffn_p = experts * 3 * h * einter if experts else 3 * h * ffn
    per_layer = attn + ffn_p + router
    embed = vocab * h
    head = 0 if c.get("tie") else vocab * h
    pos = c["max_pos"] * h if c.get("pos_embed") else 0
    total = layers * per_layer + embed + head + pos
    active_layer = attn + router + (topk * 3 * h * einter if experts else ffn_p)
    active = layers * active_layer + embed + head
    kv_bytes = 2 * layers * kh * hd * 2
    return {
        "q_dim": q_dim,
        "kv_dim": kv_dim,
        "attn": attn,
        "ffn": ffn_p,
        "router": router,
        "per_layer": per_layer,
        "embed": embed,
        "head": head,
        "pos": pos,
        "total": total,
        "active": active,
        "kv_bytes": kv_bytes,
        "kv_kib": round(kv_bytes / 1024, 2),
        "weights_q4_gb": round(total * 4.85 / 8 / 1e9, 2),
        "weights_bf16_gb": round(total * 2 / 1e9, 2),
        "gemm_flops_per_token": 2 * active,
    }


def build_anatomy() -> dict:
    here = Path(__file__).resolve().parent
    spec = json.loads((here / "anatomy.json").read_text(encoding="utf-8"))
    presets = []
    for c in spec["presets"]:
        c = dict(c)
        c["math"] = model_math(c)
        presets.append(c)
    by_id = {p["id"]: p for p in presets}
    # Sanity checks against the published sizes.
    assert abs(by_id["qwen30b"]["math"]["total"] / 1e9 - 30.5) < 0.2, by_id["qwen30b"]["math"]["total"]
    assert abs(by_id["qwen30b"]["math"]["active"] / 1e9 - 3.3) < 0.2, by_id["qwen30b"]["math"]["active"]
    assert by_id["qwen30b"]["math"]["kv_bytes"] == 96 * 1024
    assert by_id["qwen7b"]["math"]["kv_bytes"] == 56 * 1024
    assert by_id["qwen32b"]["math"]["kv_bytes"] == 256 * 1024
    sim_path = ROOT / "example" / "structure_sim.py"
    spec["simCode"] = {
        "file": "example/structure_sim.py",
        "lines": sim_path.read_text(encoding="utf-8").split("\n"),
    }
    spec["simOutput"] = run_example(sim_path)
    spec["presets"] = presets
    return spec


def run_example(path: Path) -> str:
    """Run a pure-stdlib example and return its stdout (for inline output)."""
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{path} failed ({proc.returncode}):\n{proc.stderr}")
    return proc.stdout.strip()


def build_net() -> dict:
    here = Path(__file__).resolve().parent
    spec = json.loads((here / "net.json").read_text(encoding="utf-8"))
    sim_path = ROOT / "example" / "net_sim.py"
    spec["simCode"] = {
        "file": "example/net_sim.py",
        "lines": sim_path.read_text(encoding="utf-8").split("\n"),
    }
    spec["simOutput"] = run_example(sim_path)
    return spec


def build_train() -> dict:
    here = Path(__file__).resolve().parent
    spec = json.loads((here / "train.json").read_text(encoding="utf-8"))
    code_path = ROOT / "example" / "train_tiny.py"
    spec["code"] = {
        "file": "example/train_tiny.py",
        "lines": code_path.read_text(encoding="utf-8").split("\n"),
    }
    sim_path = ROOT / "example" / "train_sim.py"
    spec["simCode"] = {
        "file": "example/train_sim.py",
        "lines": sim_path.read_text(encoding="utf-8").split("\n"),
    }
    spec["simOutput"] = run_example(sim_path)
    return spec


def build_doc_sim(name: str) -> dict:
    """Doc chapter backed by one pure-stdlib simulation script (nccl / cuda)."""
    here = Path(__file__).resolve().parent
    spec = json.loads((here / f"{name}.json").read_text(encoding="utf-8"))
    sim_rel = f"example/{name}_sim.py"
    sim_path = ROOT / sim_rel
    spec["simCode"] = {
        "file": sim_rel,
        "lines": sim_path.read_text(encoding="utf-8").split("\n"),
    }
    spec["simOutput"] = run_example(sim_path)
    return spec


def build_nccl() -> dict:
    return build_doc_sim("nccl")


def build_cuda() -> dict:
    return build_doc_sim("cuda")


def build_gpu() -> dict:
    return build_doc_sim("gpu")


def build_npu() -> dict:
    return build_doc_sim("npu")


def build_transformer() -> dict:
    return build_doc_sim("transformer")


def build_ecn() -> dict:
    return build_doc_sim("ecn")


def build_opt() -> dict:
    return build_doc_sim("opt")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def resolve_code(refs: list, ranges: dict[str, tuple[int, int]], step_id: str) -> list[dict]:
    resolved = []
    for ref in refs:
        tone = "a"
        if isinstance(ref, list):
            ref, tone = ref[0], ref[1]
        if ref not in ranges:
            raise KeyError(f"step {step_id!r}: unknown code ref {ref!r}")
        start, end = ranges[ref]
        resolved.append({"start": start, "end": end, "tone": tone, "ref": ref})
    return resolved


def build_chapter(spec_name: str = "steps.json", think_source: str = "pedagogy.json") -> dict:
    here = Path(__file__).resolve().parent
    spec = json.loads((here / spec_name).read_text(encoding="utf-8"))
    pedagogy = {}
    if think_source:
        pedagogy = json.loads((here / think_source).read_text(encoding="utf-8"))
    code_path = ROOT / spec["codeFile"]
    doc_path = ROOT / spec["docFile"]
    lines = code_path.read_text(encoding="utf-8").split("\n")
    ranges = symbol_ranges(code_path)
    sections, _ = parse_sections(doc_path)
    by_title = {sec["title"]: sec for sec in sections}

    steps = []
    for raw in spec["steps"]:
        section = by_title.get(raw["section"])
        if section is None:
            raise KeyError(f"step {raw['id']!r}: section {raw['section']!r} not found in {doc_path.name}")
        body_html = render_md(section["body"])
        if not body_html.strip():
            raise ValueError(f"step {raw['id']!r}: section {raw['section']!r} has empty body")
        steps.append(
            {
                "id": raw["id"],
                "think": pedagogy.get("stepThink", {}).get(raw["id"]),
                "title": raw["title"],
                "hint": raw.get("hint", ""),
                "sectionTitle": section["title"],
                "anchor": section["id"],
                "bodyHTML": body_html,
                "codeRanges": resolve_code(raw.get("code", []), ranges, raw["id"]),
                "viz": raw.get("viz", "none"),
                "scenario": raw.get("scenario"),
            }
        )
    return {
        "title": spec["title"],
        "codeFile": spec["codeFile"],
        "docFile": spec["docFile"],
        "lines": lines,
        "steps": steps,
    }


def build_overview() -> dict:
    sections, intro = parse_sections(ROOT / "README.md")
    why = next((s for s in sections if s["title"] == "为什么需要 KV Cache"), None)
    if why is None:
        raise KeyError("README.md: section '为什么需要 KV Cache' not found")
    lead_lines = []
    for ln in intro:
        if ln.strip().startswith("- **开始学习**"):
            break
        if ln.strip():
            lead_lines.append(ln)
    return {
        "title": "whykvc",
        "lead": render_md(lead_lines[:3]),
        "whyTitle": why["title"],
        "whyHTML": render_md(why["body"]),
    }


def _norm_text(s: str) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:'\"“”‘’（）()\[\]【】]", "", s or "")


def _find_question(qs: list[dict], needle: str) -> dict:
    for q in qs:
        if needle in q.get("q", ""):
            return q
    raise AssertionError(f"question not found: {needle}")


def _check_question_shape(label: str, qs: list[dict]) -> None:
    for i, q in enumerate(qs):
        assert isinstance(q.get("options"), list) and len(q["options"]) >= 2, f"{label} Q{i + 1}: options"
        assert 0 <= q.get("answer", -1) < len(q["options"]), f"{label} Q{i + 1}: answer index"
        assert len(set(q["options"])) == len(q["options"]), f"{label} Q{i + 1}: duplicate options"
        assert q.get("q") and q.get("why"), f"{label} Q{i + 1}: missing q/why"


def check_question_bank(data: dict) -> None:
    """Question integrity: no answer leaks, no think/quiz duplicates, balanced positions."""
    pre = data["think"]["quiz"]
    post = data["think"]["postTest"]["questions"]
    assert len(pre) == 10 and len(post) == 10, "pre/post must each have 10 questions"
    for label, qs in (("前测", pre), ("后测", post)):
        _check_question_shape(label, qs)
        for q in qs:
            assert q.get("summary") and q.get("detail"), f"{label} {q.get('id')}: missing summary/detail"
            # 「背景 / 怎么想」不能直接出现正确选项原文（避免提交前泄题）。
            correct = _norm_text(q["options"][q["answer"]])
            if len(correct) >= 12:
                assert correct not in _norm_text(q["detail"]), \
                    f"{label} {q.get('id')}: detail leaks the correct option"
        from collections import Counter
        dist = Counter(q["answer"] for q in qs)
        assert max(dist.values()) <= 4, f"{label}: answer positions too concentrated: {dist}"

    chapters = data["lessons"]
    for ch, lesson in chapters.items():
        think = lesson.get("think") or []
        quiz = lesson.get("quiz") or []
        assert len(quiz) >= 3, f"{ch}: chapter quiz needs at least 3 questions"
        _check_question_shape(f"{ch}.quiz", quiz)
        _check_question_shape(f"{ch}.think", think)
        sections = [q.get("section") for q in quiz]
        assert all(sections) and len(set(sections)) == len(sections), f"{ch}: quiz sections not distinct"
        think_text = {_norm_text(q["q"]) for q in think}
        for q in quiz:
            assert _norm_text(q["q"]) not in think_text, f"{ch}: quiz duplicates a think question"
        from collections import Counter
        dist = Counter(q["answer"] for q in quiz)
        assert len(dist) >= 2, f"{ch}: quiz answers all at the same position"


def check_question_numbers(data: dict) -> None:
    """Pin the numbers in question text to the same formulas used by the examples."""
    def option_text(q: dict) -> str:
        return q["options"][q["answer"]]

    # b2: prompt=8, n_new=7 -> prefill 8 + decode 6 = 14
    b2 = next(q for q in data["think"]["postTest"]["questions"] if q.get("id") == "b2")
    assert "8-token" in b2["q"] and "7 个" in b2["q"], "b2 question text changed"
    expected_b2 = 8 + (7 - 1)
    assert f"{expected_b2} 次" in option_text(b2), f"b2 answer should be {expected_b2}: {option_text(b2)}"
    assert str(expected_b2) in b2["why"], "b2 explanation must use 14"

    # KV: prompt=8, n_new=8 -> plain positions 92, cached 15, wasted 77
    kv_q = _find_question(data["lessons"]["kv"]["think"], "92 次位置前向")
    plain = sum(range(8, 8 + 8))
    cached = 8 + (8 - 1)
    assert plain == 92 and cached == 15 and plain - cached == 77
    for n in (str(plain), str(cached), str(plain - cached)):
        assert n in kv_q["q"], f"KV question lost {n}"
    assert "77" in kv_q["why"], "KV explanation must mention 77"

    # pre q8: 8-rank ring all-reduce, each rank sends 2(N-1)/N * S = 1.75 GiB
    ring = _find_question(data["think"]["quiz"], "8 卡做 ring")
    n, size_gib = 8, 1.0
    per_rank = 2 * (n - 1) / n * size_gib
    assert abs(per_rank - 1.75) < 1e-9
    assert "1.75" in option_text(ring), f"ring answer should mention 1.75: {option_text(ring)}"
    assert "1.75" in ring["why"]

    # NCCL: ring all-reduce 2(N-1) steps; N=8 -> 14
    nccl_steps = _find_question(data["lessons"]["nccl"]["think"], "ring all-reduce 一共要走多少步")
    assert 2 * (8 - 1) == 14 and "14" in option_text(nccl_steps)

    # NCCL: per-rank bytes for N=8, S=128 MiB -> 224 MiB
    nccl_bytes = _find_question(data["lessons"]["nccl"]["quiz"], "每卡大约要发出多少字节")
    per_rank_mib = 2 * (8 - 1) / 8 * 128
    assert per_rank_mib == 224 and "224" in option_text(nccl_bytes)

    # CUDA: 32 threads x 4 B contiguous = 128 B = 4 x 32 B sectors
    cuda_sectors = _find_question(data["lessons"]["cuda"]["think"], "32 个线程各读 4 B")
    assert 32 * 4 / 32 == 4 and "4 个" in option_text(cuda_sectors)

    # Train: BF16 weight 2 + grad 2 + AdamW 12 = 16 B/param
    train_bpp = _find_question(data["lessons"]["train"]["think"], "参数状态大约占多少 B/参数")
    assert 2 + 2 + 12 == 16 and "16 B" in option_text(train_bpp)


def check_sim_units(data: dict) -> None:
    """The generated sim stdout must use decimal GB, not mislabeled GiB."""
    for ch in ("gpu", "train", "transformer"):
        out = data[ch].get("simOutput", "")
        assert "GB" in out, f"{ch}: simOutput should use GB"
        assert "GiB" not in out, f"{ch}: simOutput still contains GiB"
    assert "CPU 桌面" in data["gpu"]["simOutput"] and "CPU 服务器" in data["gpu"]["simOutput"]
    assert "fit 72GB" in data["train"]["simOutput"]


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    pedagogy = json.loads((here / "pedagogy.json").read_text(encoding="utf-8"))
    data = {
        "overview": build_overview(),
        "kv": build_chapter(),
        "prefix": build_chapter("steps.prefix.json", think_source=None),
        "paged": build_chapter("steps.paged.json", think_source=None),
        "think": {"quiz": pedagogy["quiz"], "postTest": pedagogy["postTest"]},
        "attribution": pedagogy["attribution"],
        "reflect": pedagogy["reflect"],
        "lessons": pedagogy["lessons"],
        "anatomy": build_anatomy(),
        "train": build_train(),
        "net": build_net(),
        "nccl": build_nccl(),
        "cuda": build_cuda(),
        "ecn": build_ecn(),
        "gpu": build_gpu(),
        "npu": build_npu(),
        "transformer": build_transformer(),
        "opt": build_opt(),
    }
    for ch in ("anatomy", "net", "train", "nccl", "cuda", "ecn", "gpu", "npu", "transformer", "opt"):
        data["lessons"][ch] = data[ch]["lesson"]
    check_question_bank(data)
    check_question_numbers(data)
    check_sim_units(data)
    (DATA / "data.js").write_text(
        "window.LK_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8"
    )
    trace = collect_traces.build()
    prefix = collect_traces.prefix_trace()
    paged = collect_traces.paged_trace()
    (DATA / "traces.js").write_text(
        "window.LK_TRACES = "
        + json.dumps({"kv": trace, "prefix": prefix, "paged": paged}, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    kv = data["kv"]
    print(f"data.js   : {len(kv['lines'])} lines, {len(kv['steps'])} steps")
    print(f"traces.js : kv source={trace['source']} steps={len(trace['steps'])} · "
          f"prefix scenarios={len(prefix['scenarios'])}")
    doc = [f"{ch}:{len(build['lesson']['think'])} think / {len(build['simOutput'].splitlines())} out"
           for ch, build in (("anatomy", data["anatomy"]), ("nccl", data["nccl"]), ("cuda", data["cuda"]),
                             ("ecn", data["ecn"]), ("gpu", data["gpu"]),
                             ("npu", data["npu"]), ("transformer", data["transformer"]), ("opt", data["opt"]))]
    print("doc chapters:", " · ".join(doc))
    print("built:", ", ".join(s["id"] for s in kv["steps"]))


if __name__ == "__main__":
    main()
