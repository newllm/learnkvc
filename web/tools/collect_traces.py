#!/usr/bin/env python3
"""Generate web/data/traces.js for the interactive reader.

The counters are the *same bookkeeping* that ``example/kv_cache.py`` checks in
``print_summary()``: input lengths and (query, key) pairs per layer.  When
PyTorch is importable we run the real demo and take the real tokens/logits;
otherwise we fall back to the analytic trace (documented token sequence from
the README run + the exact formulas asserted in ``print_summary``).

Run:  python3 web/tools/collect_traces.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "example"
OUT = ROOT / "web" / "data" / "traces.js"

PROMPT = [7, 12, 3, 25, 9, 18, 1, 30]
#: Token sequence produced by the demo run documented in README.md
#: ("生成结果：[30, 15, 15, 15, 41, 41, 41, 41]").
GENERATED = [30, 15, 15, 15, 41, 41, 41, 41]

MODEL = {
    "layers": 2,
    "q_heads": 4,
    "kv_heads": 2,
    "head_dim": 8,
    "d_model": 32,
    "vocab": 64,
    "max_pos": 256,
    "seed": 13,
}

NUMERICS_DIFF = 4.47e-08  # README: measured max |plain - cached| logits diff


def _step(trace_steps: list[dict], s: int, plain_input: int, cached_input: int,
          plain_pos: int, cached_pos: int, plain_pairs: int, cached_pairs: int,
          cache_len: int, token: int) -> None:
    trace_steps.append(
        {
            "s": s,
            "token": token,
            "plain_input": plain_input,
            "cached_input": cached_input,
            "plain_pos": plain_pos,
            "cached_pos": cached_pos,
            "plain_pairs": plain_pairs,
            "cached_pairs": cached_pairs,
            "cache_len": cache_len,
            "wasted": plain_pos - cached_pos,
        }
    )


def analytic_trace() -> dict:
    n_layers = MODEL["layers"]
    prompt_len, n_new = len(PROMPT), len(GENERATED)
    steps: list[dict] = []
    plain_pos = cached_pos = plain_pairs = cached_pairs = 0
    for s in range(n_new):
        plain_input = prompt_len + s
        cached_input = prompt_len if s == 0 else 1
        past = 0 if s == 0 else prompt_len + s - 1
        plain_pos += plain_input
        cached_pos += cached_input
        plain_pairs += n_layers * (plain_input * (plain_input + 1) // 2)
        cached_pairs += n_layers * (
            cached_input * past + cached_input * (cached_input + 1) // 2
        )
        _step(steps, s, plain_input, cached_input, plain_pos, cached_pos,
              plain_pairs, cached_pairs, prompt_len + s, GENERATED[s])
    return {
        "source": "analytic",
        "note": "PyTorch 未安装：token 序列取自 README 的示例输出，计数按 print_summary 的断言公式生成。",
        "model": MODEL,
        "prompt": PROMPT,
        "generated": GENERATED,
        "steps": steps,
        "totals": {
            "plain_pos": plain_pos,
            "cached_pos": cached_pos,
            "plain_pairs": plain_pairs,
            "cached_pairs": cached_pairs,
            "wasted": plain_pos - cached_pos,
        },
        "numerics": {"diff": NUMERICS_DIFF, "threshold": 1e-05},
        "features": features(),
    }


def reuse_data() -> dict:
    """Multi-turn and multi-request reuse bookkeeping (policy-level trace)."""
    system, user, answer, question = 200, 50, 100, 50
    token_kib = 96  # Qwen3-Coder-30B-A3B, fp16 KV

    def gib(tokens: int) -> float:
        return tokens * token_kib / 1024 / 1024

    turns = []
    for i in range(1, 6):
        prompt = system + (i - 1) * (user + answer) + user
        prefill_cache = system + user if i == 1 else user
        kv_tokens = prompt + answer
        turns.append(
            {
                "i": i,
                "prompt": prompt,
                "prefill_nocache": prompt,
                "prefill_cache": prefill_cache,
                "kv_tokens": kv_tokens,
                "kv_gib": round(gib(kv_tokens), 3),
            }
        )

    requests = []
    for j in range(1, 6):
        prefill_cache = system + question if j == 1 else question
        requests.append(
            {
                "i": j,
                "prefill_nocache": system + question,
                "prefill_cache": prefill_cache,
                "kv_nocache": system + question,
                "kv_private": question,
                "kv_gib_nocache": round(gib(system + question), 3),
                "kv_gib_private": round(gib(question), 3),
            }
        )

    return {
        "token_kib": token_kib,
        "system": system,
        "user": user,
        "answer": answer,
        "question": question,
        "turns": turns,
        "requests": requests,
    }


def features() -> dict:
    return {
        "gqa": {
            "q_heads": 4,
            "kv_heads": 2,
            "kv_shape": [1, 2, 3, 8],
            "mapping": [[0, 0], [1, 0], [2, 1], [3, 1]],
        },
        "window": {
            "feed": 6,
            "max_length": 4,
            "length": 4,
            "dropped": 2,
            "kept_positions": [2, 3, 4, 5],
        },
        "reorder": {"batch": 2, "positions": 3, "indices": [1, 0]},
        "fork": {"parent_len": 2, "branch_len": 3, "extra": 1},
    }


def torch_trace() -> dict:
    """Run the real demo (requires torch).  Raises on any incompatibility."""
    import torch  # noqa: F401

    sys.path.insert(0, str(EXAMPLES))
    import kv_cache  # type: ignore

    model = kv_cache.TinyModel(seed=kv_cache.SEED)
    prompt = list(PROMPT)
    n_new = len(GENERATED)

    plain_stats = kv_cache.Stats()
    plain_seq, plain_steps = kv_cache.generate(
        model, prompt, n_new, use_cache=False, stats=plain_stats
    )
    cached_stats = kv_cache.Stats()
    cached_seq, cached_steps = kv_cache.generate(
        model, prompt, n_new, use_cache=True, stats=cached_stats
    )
    assert plain_seq == cached_seq

    steps: list[dict] = []
    plain_pos = cached_pos = plain_pairs = cached_pairs = 0
    cache_len = 0
    for s, (pstep, cstep) in enumerate(zip(plain_steps, cached_steps)):
        plain_pos += pstep.input_tokens
        cached_pos += cstep.input_tokens
        plain_pairs += MODEL["layers"] * (
            pstep.input_tokens * (pstep.input_tokens + 1) // 2
        )
        past = 0 if s == 0 else prompt.__len__() + s - 1
        cached_pairs += MODEL["layers"] * (
            cstep.input_tokens * past + cstep.input_tokens * (cstep.input_tokens + 1) // 2
        )
        if s == 0:
            cache_len = pstep.input_tokens
        else:
            cache_len += cstep.input_tokens
        _step(steps, s, pstep.input_tokens, cstep.input_tokens, plain_pos, cached_pos,
              plain_pairs, cached_pairs, cache_len, pstep.token)

    # Numerics: one full forward vs prefill + decode, like check_numerics().
    token = 33
    plain_logits = model.forward(torch.tensor(prompt + [token]))
    cache = kv_cache.KVCache(model.num_layers)
    model.forward(torch.tensor(prompt), cache=cache)
    cached_logits = model.forward(torch.tensor([token]), cache=cache)
    diff = float((plain_logits - cached_logits).abs().max().item())
    assert int(plain_logits.argmax()) == int(cached_logits.argmax())

    return {
        "source": f"torch {torch.__version__}",
        "note": "由 example/kv_cache.py 的真实运行生成。",
        "model": MODEL,
        "prompt": prompt,
        "generated": plain_seq[len(prompt):],
        "steps": steps,
        "totals": {
            "plain_pos": plain_pos,
            "cached_pos": cached_pos,
            "plain_pairs": plain_pairs,
            "cached_pairs": cached_pairs,
            "wasted": plain_pos - cached_pos,
        },
        "numerics": {"diff": diff, "threshold": 1e-05},
        "features": features(),
    }


def build() -> dict:
    try:
        trace = torch_trace()
    except Exception as exc:  # torch missing or demo mismatch: use analytic
        trace = analytic_trace()
        trace["note"] += f"（torch 路径失败：{type(exc).__name__}: {exc}）"
    trace["reuse"] = reuse_data()
    # Sanity: the analytic/real trace must match the values asserted by the demo.
    assert trace["totals"] == {
        "plain_pos": 92,
        "cached_pos": 15,
        "plain_pairs": 1192,
        "cached_pairs": 240,
        "wasted": 77,
    }, trace["totals"]
    return trace


def prefix_trace() -> dict:
    """Run example/prefix_cache_demo.py (pure stdlib) and snapshot the block pool
    after every admit/decode phase, for the interactive chapter."""
    import importlib.util

    mod_name = "whykvc_prefix_demo"
    spec = importlib.util.spec_from_file_location(mod_name, EXAMPLES / "prefix_cache_demo.py")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves ``cls.__module__`` through sys.modules; register the
    # freshly loaded module before executing it or @dataclass raises.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

    class TraceSim(mod.Simulator):
        def __init__(self, cache, requests):
            super().__init__(cache, requests)
            self.snaps = []

        def snapshot(self, step, phase):
            blocks = []
            for b in sorted(self.cache.blocks.values(), key=lambda x: x.block_id):
                blocks.append(
                    {
                        "id": b.block_id,
                        "hash": format(b.block_hash & 0xFFFF, "04x"),
                        "tokens": list(b.tokens),
                        "ref": b.ref_count,
                        "last": b.last_used,
                        "children": len(b.children),
                        "label": b.label,
                        "parent": format(b.parent_hash & 0xFFFF, "04x") if b.parent_hash else None,
                    }
                )
            self.snaps.append(
                {
                    "step": step,
                    "phase": phase,
                    "active": list(self._active),
                    "pending": [r.request_id for r in self._pending],
                    "blocks": blocks,
                    "trace": self.trace[-8:],
                    "clock": self.cache._clock,
                }
            )

        def _admit(self, step):
            super()._admit(step)
            self.snapshot(step, "admit")

        def _decode(self, step):
            super()._decode(step)
            self.snapshot(step, "decode")

        def run(self):
            step = 0
            while self._pending or self._active:
                self._admit(step)
                self._decode(step)
                step += 1
                if step > 1000:
                    raise RuntimeError("simulation did not terminate")
            return self.records

    def run_scenario(title, capacity, requests):
        cache = mod.PrefixCache(capacity_blocks=capacity)
        sim = TraceSim(cache, requests)
        records = sim.run()
        st = cache.stats
        return {
            "title": title,
            "capacity": capacity,
            "block_size": mod.BLOCK_SIZE,
            "requests": [
                {
                    "id": r.request_id,
                    "prompt": len(r.request.prompt),
                    "cached": r.cached_tokens,
                    "prefill": r.prefill_tokens,
                    "blocks": list(r.block_labels),
                    "hit": round(r.hit_rate, 3),
                    "arrival": r.request.arrival,
                }
                for r in sorted(records.values(), key=lambda x: (x.request.arrival, x.request_id))
            ],
            "snapshots": sim.snaps,
            "stats": {
                "prompt_tokens": st.prompt_tokens,
                "reused_tokens": st.reused_tokens,
                "prefilled_tokens": st.prefilled_tokens,
                "evicted": st.blocks_evicted,
                "peak_blocks": st.peak_blocks,
                "peak_in_use": st.peak_blocks_in_use,
                "hit_rate": round(st.hit_rate, 3),
            },
        }

    shared = run_scenario(
        "共享 system prompt",
        64,
        [
            mod.req("req-refund-1", 0, mod.SYSTEM_PROMPT + tuple(range(201, 209))),
            mod.req("req-shipping-2", 0, mod.SYSTEM_PROMPT + tuple(range(211, 217))),
            mod.req("req-tech-3", 1, mod.SYSTEM_PROMPT + mod.FEW_SHOT + tuple(range(221, 226))),
            mod.req("req-tech-4", 2, mod.SYSTEM_PROMPT + mod.FEW_SHOT + tuple(range(221, 226))),
            mod.req("req-billing-5", 3, mod.ALT_SYSTEM_PROMPT + tuple(range(231, 243))),
        ],
    )
    prefix_a = tuple(range(10, 34))
    prefix_c = tuple(range(40, 64))
    eviction = run_scenario(
        "块池不足 · LRU 淘汰",
        10,
        [
            mod.req("req-a", 0, prefix_a + tuple(range(70, 78)), max_new_tokens=1),
            mod.req("req-b", 1, prefix_a + tuple(range(80, 88)), max_new_tokens=1),
            mod.req("req-c", 2, prefix_c + tuple(range(90, 106))),
            mod.req("req-d", 4, prefix_a + tuple(range(70, 78))),
        ],
    )
    phrase = (9, 9, 9, 9)
    prefix = tuple(range(600, 612))
    middle = tuple(range(620, 628))
    repeated = run_scenario(
        "相同 4-token，不同位置",
        32,
        [
            mod.req("req-x", 0, prefix + phrase + middle + phrase, max_new_tokens=1),
            mod.req("req-y", 1, prefix + phrase + tuple(range(640, 644)), max_new_tokens=1),
        ],
    )
    return {
        "source": "example/prefix_cache_demo.py（真实运行）",
        "scenarios": {"shared": shared, "eviction": eviction, "repeated": repeated},
    }


def paged_trace() -> dict:
    """Replay example/paged_attention.py's main() bookkeeping without torch.

    Block allocation, block tables, release and internal waste are pure
    bookkeeping; the attention assertion is documented but not re-run here."""
    import random

    block_size = 4
    num_blocks = 8
    free = list(range(num_blocks))
    random.Random(7).shuffle(free)
    owner = {i: None for i in range(num_blocks)}
    used = {i: 0 for i in range(num_blocks)}
    tables = {"seq-a": [], "seq-b": [], "seq-c": []}
    lengths = {"seq-a": 0, "seq-b": 0, "seq-c": 0}
    snaps = []

    def snapshot(label, event, seq, tokens):
        snaps.append(
            {
                "label": label,
                "event": event,
                "seq": seq,
                "tokens": tokens,
                "free": list(free),
                "blocks": [
                    {"id": i, "owner": owner[i], "used": used[i]} for i in range(num_blocks)
                ],
                "tables": {k: list(v) for k, v in tables.items()},
                "lengths": dict(lengths),
                "waste": sum(
                    len(tables[k]) * block_size - lengths[k] for k in ("seq-b", "seq-c")
                ),
            }
        )

    def feed(seq, n):
        for _ in range(n):
            slot = lengths[seq] % block_size
            if slot == 0:
                bid = free.pop()
                tables[seq].append(bid)
                owner[bid] = seq
                used[bid] = 0
            bid = tables[seq][-1]
            used[bid] += 1
            lengths[seq] += 1
        snapshot("feed " + seq + " · " + str(n) + " tok", "feed", seq, n)

    snapshot("初始：8 个空闲块（已打乱顺序）", "init", None, 0)
    feed("seq-a", 10)
    feed("seq-b", 6)
    feed("seq-a", 2)
    feed("seq-a", 2)
    feed("seq-b", 1)

    released = list(tables["seq-a"])
    for bid in released:
        owner[bid] = None
        used[bid] = 0
        free.append(bid)
    tables["seq-a"] = []
    lengths["seq-a"] = 0
    snapshot("free seq-a · 释放 " + str(len(released)) + " 块", "free", "seq-a", len(released))

    feed("seq-c", 5)
    reused = sorted(set(released) & set(tables["seq-c"]))
    snapshot(
        "feed seq-c · 5 tok（复用 " + str(len(reused)) + " 块）",
        "reuse",
        "seq-c",
        len(reused),
    )

    return {
        "block_size": block_size,
        "num_blocks": num_blocks,
        "q_heads": 4,
        "kv_heads": 2,
        "head_dim": 8,
        "seed": 7,
        "sequences": ["seq-a", "seq-b", "seq-c"],
        "snapshots": snaps,
        "released": released,
        "reused_by_c": reused,
        "total_waste": sum(
            len(tables[k]) * block_size - lengths[k] for k in ("seq-b", "seq-c")
        ),
    }


def main() -> None:
    trace = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(trace, ensure_ascii=False, indent=1)
    OUT.write_text("window.LK_TRACES = {" + json.dumps({"kv": trace}, ensure_ascii=False) + "};\n")
    print(f"wrote {OUT.relative_to(ROOT)}  source={trace['source']}  "
          f"steps={len(trace['steps'])}")


if __name__ == "__main__":
    main()
