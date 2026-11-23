#!/usr/bin/env python3
"""Simulate NCCL ring collectives and PD-separated KV transfer.

Run:
    python3 example/net_sim.py

Pure standard library: no GPU, no NCCL, no torch. The point is to make the
arithmetic visible: ring steps, bytes per rank, time, bus bandwidth, and the
KV-transfer part of prefill/decode disaggregation (PD).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MIB = 1 << 20
GIB = 1 << 30


@dataclass(frozen=True)
class Link:
    """One network hop. gbps is the port rate, latency is per hop."""

    gbps: float = 400.0
    latency_us: float = 2.0

    @property
    def bytes_per_s(self) -> float:
        return self.gbps / 8 * 1e9

    @property
    def latency_s(self) -> float:
        return self.latency_us / 1e6


def ring_schedule(n: int, op: str) -> list[tuple[str, list[tuple[int, object, int]]]]:
    """Return one row per ring step: (phase, [(rank, chunk, destination)])."""
    rows = []
    if op == "allreduce":
        for s in range(2 * (n - 1)):
            if s < n - 1:
                phase, local = "reduce-scatter", s
                chunk = lambda r, local=local: (r - local) % n
            else:
                phase, local = "all-gather", s - (n - 1)
                chunk = lambda r, local=local: (r - local + 1) % n
            rows.append((phase, [(r, chunk(r), (r + 1) % n) for r in range(n)]))
    elif op in ("allgather", "reducescatter"):
        phase = "all-gather" if op == "allgather" else "reduce-scatter"
        for s in range(n - 1):
            rows.append((phase, [(r, (r - s) % n, (r + 1) % n) for r in range(n)]))
    elif op == "alltoall":
        for s in range(n - 1):
            rows.append(("all-to-all", [(r, (r + s + 1) % n, (r + 1) % n) for r in range(n)]))
    else:
        rows.append(("send/recv", [(0, "S", 1)]))
    return rows


def ring_stats(n: int, size: int, link: Link, op: str = "allreduce") -> dict:
    """Theoretical ring time and bytes for one collective."""
    if op == "sendrecv":
        steps = 1
        per_rank = size
        seconds = link.latency_s + size / link.bytes_per_s
        chunk = size
    else:
        steps = 2 * (n - 1) if op == "allreduce" else n - 1
        chunk = size / n
        seconds = steps * (link.latency_s + chunk / link.bytes_per_s)
        ratio = 2 * (n - 1) / n if op == "allreduce" else (n - 1) / n
        per_rank = size * ratio
    return {
        "steps": steps,
        "chunk": chunk,
        "per_rank": per_rank,
        "network": per_rank * n,
        "seconds": seconds,
        "busbw": per_rank / seconds if seconds else 0.0,
    }


def tree_seconds(n: int, size: int, link: Link) -> float:
    """Approximate all-reduce tree: 2*log2(N) steps, whole S per step."""
    return 2 * math.ceil(math.log2(n)) * (link.latency_s + size / link.bytes_per_s)


@dataclass(frozen=True)
class PDConfig:
    prompt: int = 4096
    output: int = 128
    kv_per_token: int = 96 * 1024  # Qwen3-30B-A3B, FP16
    layers: int = 48
    q_heads: int = 32
    head_dim: int = 128
    active_params: float = 3.35e9
    prefill_tflops: float = 165.0
    decode_gbps: float = 1008.0
    prefill_gpus: int = 1
    decode_gpus: int = 1


def pd_row(cfg: PDConfig, link: Link) -> dict:
    kv = cfg.prompt * cfg.kv_per_token
    transfer_s = link.latency_s + kv / link.bytes_per_s
    prefill_flops = 2 * cfg.active_params * cfg.prompt + 4 * cfg.layers * cfg.q_heads * cfg.head_dim * cfg.prompt**2
    prefill_s = prefill_flops / (cfg.prefill_tflops * 1e12 * cfg.prefill_gpus)
    decode_bytes = cfg.active_params * 2 + kv
    tpot_s = decode_bytes / (cfg.decode_gbps * 1e9 * cfg.decode_gpus)
    ttft_s = prefill_s + transfer_s + tpot_s
    total_s = prefill_s + transfer_s + cfg.output * tpot_s
    return {
        "kv": kv,
        "transfer_ms": transfer_s * 1e3,
        "prefill_ms": prefill_s * 1e3,
        "ttft_ms": ttft_s * 1e3,
        "tpot_ms": tpot_s * 1e3,
        "total_s": total_s,
        "link_req_s": link.bytes_per_s / kv,
    }


def main() -> None:
    link = Link(gbps=400, latency_us=2)
    size = 128 * MIB

    print("NCCL ring: S=128 MiB, 400 Gb/s (50 GB/s), 2 us/hop")
    print(f"{'N':>2} {'steps':>5} {'chunk MiB':>9} {'rank MiB':>9} {'net MiB':>9} {'ring ms':>8} {'busbw GB/s':>10} {'tree ms':>8}")
    for n in (2, 4, 8, 16):
        st = ring_stats(n, size, link)
        print(
            f"{n:>2} {st['steps']:>5} {st['chunk']/MIB:>9.1f} {st['per_rank']/MIB:>9.1f} "
            f"{st['network']/MIB:>9.1f} {st['seconds']*1e3:>8.2f} {st['busbw']/1e9:>10.2f} "
            f"{tree_seconds(n, size, link)*1e3:>8.2f}"
        )

    print("\nring all-reduce schedule, N=4 (every rank sends one chunk to the next)")
    print(f"{'step':>4} {'phase':<14} sends")
    rows = ring_schedule(4, "allreduce")
    for i, (phase, sends) in enumerate(rows, 1):
        text = "  ".join(f"{r}:c{c}->{d}" for r, c, d in sends)
        print(f"{i:>4} {phase:<14} {text}")

    print("\nPD separation: prefill -> KV transfer -> decode, 400 Gb/s, 2 us")
    print(f"{'prompt':>6} {'KV GiB':>7} {'xfer ms':>8} {'prefill ms':>10} {'TTFT ms':>8} {'TPOT ms':>8} {'total s':>8} {'link req/s':>10}")
    for prompt in (2048, 8192, 32768):
        row = pd_row(PDConfig(prompt=prompt), link)
        print(
            f"{prompt:>6} {row['kv']/GIB:>7.2f} {row['transfer_ms']:>8.1f} {row['prefill_ms']:>10.0f} "
            f"{row['ttft_ms']:>8.0f} {row['tpot_ms']:>8.1f} {row['total_s']:>8.2f} {row['link_req_s']:>10.0f}"
        )

    # Assertions make this a simulation, not a pile of prints.
    st = ring_stats(4, size, link)
    assert st["steps"] == 6
    assert abs(st["per_rank"] - size * 1.5) < 1
    assert abs(st["busbw"] - st["per_rank"] / st["seconds"]) < 1
    assert pd_row(PDConfig(prompt=8192), link)["ttft_ms"] > pd_row(PDConfig(prompt=8192), link)["transfer_ms"]
    assert pd_row(PDConfig(prompt=32768), link)["transfer_ms"] > pd_row(PDConfig(prompt=2048), link)["transfer_ms"]
    print("\nOK: ring steps/bytes and PD transfer scale as expected.")


if __name__ == "__main__":
    main()
