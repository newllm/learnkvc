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
    return {
        "quiz": spec["quiz"],
        "presets": presets,
        "timeline": spec["timeline"],
        "gpuLoad": spec["gpuLoad"],
        "kernels": spec["kernels"],
        "notes": spec["notes"],
        "context": spec["context"],
        "files": spec["files"],
        "runtime": spec["runtime"],
        "gpuFunctions": spec["gpuFunctions"],
    }


def build_net() -> dict:
    here = Path(__file__).resolve().parent
    return json.loads((here / "net.json").read_text(encoding="utf-8"))


def build_train() -> dict:
    here = Path(__file__).resolve().parent
    spec = json.loads((here / "train.json").read_text(encoding="utf-8"))
    code_path = ROOT / "example" / "train_tiny.py"
    spec["code"] = {
        "file": "example/train_tiny.py",
        "lines": code_path.read_text(encoding="utf-8").split("\n"),
    }
    return spec


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
    return {
        "title": "learnkvc",
        "lead": render_md([ln for ln in intro if ln.strip()][:3]),
        "whyTitle": why["title"],
        "whyHTML": render_md(why["body"]),
    }


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
        "homework": pedagogy["homework"],
        "anatomy": build_anatomy(),
        "train": build_train(),
        "net": build_net(),
    }
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
    print("built:", ", ".join(s["id"] for s in kv["steps"]))


if __name__ == "__main__":
    main()
