#!/usr/bin/env python3
"""Simulate ECN marking + DCQCN rate control + PFC pause for lossless RoCEv2.

Run:
    python3 example/ecn_sim.py

Pure standard library: no switch, no NIC, no RDMA. One simplified switch queue
(入队 R, 出队 = 链路带宽), WRED-style ECN marking, delayed CNP feedback, DCQCN
alpha/rate update, and PFC XOFF/XON pauses. Three scenarios compare
PFC-only / ECN+DCQCN / ECN+DCQCN+PFC, followed by an ECN threshold sweep.

Numbers are normalized (量级/简化 LLM): one tick ~ 10 us, queue/rate in MB per
tick, link rate 10 MB/tick. Every printed number is produced below and checked
by the assertions at the end.
"""

from __future__ import annotations

from dataclasses import dataclass

TICKS = 2000
LINK_MB = 10.0          # dequeue per tick (line rate)
PKT_MB = 0.5            # packet size in the model
BUF_MB = 20.0           # switch buffer
XOFF_MB = 6.0           # PFC pause when standing queue >= XOFF
XON_MB = 3.0            # resume when queue <= XON
PAUSE_TICKS = 1         # one pause event stops the sender for 1 tick
CNP_RTT = 6             # CE mark -> CNP reaches sender after 6 ticks
G = 1 / 16              # DCQCN alpha weight
RAI_MB = 0.15           # additive increase per tick (no CNP)
R_MIN_MB = 8.5          # rate floor
R_MAX_MB = 12.5         # rate ceiling
CNP_INTERVAL = 2        # at most one CNP per 2 ticks (receiver rate limit)
DEFAULT_MAX_P = 0.2     # WRED max marking probability
SEED = 12345


@dataclass(frozen=True)
class Scenario:
    name: str
    pfc: bool
    ecn: bool
    emin: float
    emax: float
    max_p: float = DEFAULT_MAX_P


SCENARIOS = (
    Scenario("A 只有 PFC", pfc=True, ecn=False, emin=0.0, emax=0.0),
    Scenario("B ECN+DCQCN", pfc=False, ecn=True, emin=2.0, emax=5.0),
    Scenario("C ECN+DCQCN+PFC", pfc=True, ecn=True, emin=2.0, emax=5.0),
)

SCAN = ((1.0, 3.0), (3.0, 6.0), (4.0, 8.0), (6.0, 12.0))


def _lcg(seed: int):
    """Deterministic 64-bit LCG; the same seed always marks the same packets."""
    state = seed

    def rand() -> float:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (state >> 33) / (1 << 33)

    return rand


def mark_probability(q: float, emin: float, emax: float, max_p: float) -> float:
    """WRED: below min never mark; [min, max) linear; >= max mark everything."""
    if emax <= emin:
        return 0.0
    if q < emin:
        return 0.0
    if q >= emax:
        return 1.0
    return (q - emin) / (emax - emin) * max_p


def simulate(scn: Scenario) -> dict:
    rng = _lcg(SEED)
    queue = 0.0
    rate = R_MAX_MB                 # offered load starts above line rate
    alpha = 0.0
    paused = False
    pause_left = 0
    pause_events = 0
    paused_ticks = 0
    drops = 0.0
    delivered = 0.0
    sent_pkts = 0
    marked_pkts = 0
    queue_sum = 0.0
    rate_sum = 0.0
    cnp_due = [0] * (TICKS + CNP_RTT + 2)
    last_cnp = -10**9
    pkt_accum = 0.0
    pause_log: list[tuple[int, float, float]] = []

    for t in range(TICKS):
        # Receiver's CNP arrives after CNP_RTT; one CNP updates alpha and cuts rate.
        if scn.ecn and cnp_due[t]:
            alpha = (1 - G) * alpha + G
            rate = max(R_MIN_MB, rate * (1 - alpha / 2))
        elif scn.ecn:
            alpha = (1 - G) * alpha
            if not (scn.pfc and paused):
                rate = min(R_MAX_MB, rate + RAI_MB)

        send = 0.0 if (scn.pfc and paused) else rate
        q_ecn = queue                      # standing queue used by WRED
        space = BUF_MB - queue
        if send >= space:
            drops += send - space
            enq = space
        else:
            enq = send
        queue += enq

        pkt_accum += enq
        marks = 0
        while pkt_accum + 1e-12 >= PKT_MB:
            pkt_accum -= PKT_MB
            sent_pkts += 1
            if scn.ecn:
                p = mark_probability(q_ecn, scn.emin, scn.emax, scn.max_p)
                if rng() < p:
                    marks += 1
                    marked_pkts += 1
        if marks and t - last_cnp >= CNP_INTERVAL:
            cnp_due[min(t + CNP_RTT, len(cnp_due) - 1)] = 1
            last_cnp = t

        served = min(queue, LINK_MB)       # dequeue = link bandwidth
        queue -= served
        delivered += served
        queue_sum += queue
        rate_sum += send

        # PFC: pause the sender at XOFF; resume at XON, extend if still above.
        if scn.pfc:
            if paused:
                paused_ticks += 1
                pause_left -= 1
                if pause_left <= 0:
                    if queue <= XON_MB:
                        paused = False
                    else:
                        pause_left = PAUSE_TICKS
            elif queue >= XOFF_MB:
                paused = True
                pause_left = PAUSE_TICKS
                pause_events += 1
                pause_log.append((t, queue, rate))

    return {
        "name": scn.name,
        "pfc": scn.pfc,
        "ecn": scn.ecn,
        "drops": drops,
        "throughput": delivered / TICKS,
        "avg_queue": queue_sum / TICKS,
        "mark_rate": marked_pkts / max(1, sent_pkts),
        "pause_events": pause_events,
        "paused_ticks": paused_ticks,
        "avg_rate": rate_sum / TICKS,
        "final_rate": rate,
        "pause_log": pause_log,
    }


def scenario_row(r: dict) -> str:
    return (f"{r['name']:<22}{('on' if r['pfc'] else 'off'):>4}{('on' if r['ecn'] else 'off'):>5}"
            f"{r['drops']:>10.1f}{r['throughput']:>10.2f}{r['avg_queue']:>9.2f}"
            f"{r['mark_rate'] * 100:>8.1f}%{r['pause_events']:>9}{r['paused_ticks']:>8}")


def main() -> None:
    print("ECN + DCQCN + PFC ledger (简化 LLM；归一化单位，1 tick ≈ 10 us 量级)")
    print(f"params: C={LINK_MB:.0f} MB/tick, R0={R_MAX_MB:.1f}, R_MIN={R_MIN_MB:.1f}, "
          f"PKT={PKT_MB:.1f} MB, BUF={BUF_MB:.0f} MB")
    print(f"        ECN min/max per case, max_p={DEFAULT_MAX_P}; PFC XOFF={XOFF_MB:.0f}, "
          f"XON={XON_MB:.0f}, pause={PAUSE_TICKS} tick; RTT={CNP_RTT}, g=1/16, RAI={RAI_MB}")

    # ---- 1. three scenarios --------------------------------------------- #
    results = [simulate(scn) for scn in SCENARIOS]
    a, b, c = results
    print(f"\n[1] scenario comparison (T={TICKS} ticks, mark-rate uses ECN (min=2,max=5))")
    print(f"{'scenario':<22}{'PFC':>4}{'ECN':>5}{'drop MB':>10}{'thr MB/t':>10}"
          f"{'avgQ MB':>9}{'mark%':>9}{'pause#':>9}{'paused':>8}")
    for r in results:
        print(scenario_row(r))
    print(f"note: A 的 pause 把链路打断，吞吐 {a['throughput']:.2f}；C 用 ECN 提前降速，pause 少 "
          f"{a['pause_events'] / max(1, c['pause_events']):.1f}x、吞吐 {c['throughput']:.2f}；"
          f"B 无 PFC 时靠 buffer 吸收，仍溢出丢包 {b['drops']:.1f} MB（重传代价未建模）。")

    # ---- 2. ECN threshold sweep ----------------------------------------- #
    print("\n[2] ECN threshold scan (ECN+DCQCN+PFC, same seed)")
    print(f"{'min/max':>9}{'mark%':>8}{'avgQ MB':>9}{'thr MB/t':>10}{'pause#':>8}  reading")
    readings = ("早标记/降速早", "偏晚", "晚标记/PFC 变多", "max > XOFF，退化为只有 PFC")
    scan_rows = []
    for (emin, emax), reading in zip(SCAN, readings):
        r = simulate(Scenario(f"scan {emin:.0f}/{emax:.0f}", pfc=True, ecn=True,
                              emin=emin, emax=emax))
        scan_rows.append(r)
        print(f"{emin:>5.0f}/{emax:<3.0f}{r['mark_rate'] * 100:>8.1f}{r['avg_queue']:>9.2f}"
              f"{r['throughput']:>10.2f}{r['pause_events']:>8}  {reading}")

    # ---- 3. PFC pause trace (scenario A) -------------------------------- #
    print("\n[3] PFC pause trace (scenario A, first cycles)")
    print(f"{'tick':>5}{'queue MB':>10}{'rate MB/t':>11}  event")
    for t, q, rate in a["pause_log"][:4]:
        print(f"{t:>5}{q:>10.2f}{rate:>11.2f}  XOFF -> pause {PAUSE_TICKS} tick")

    # ---- 4. inference link ---------------------------------------------- #
    print("\n[4] inference link: RoCEv2 上的 TP all-reduce（decode）或 PD KV 一旦丢包，")
    print("    Go-Back-N 重传会放大延迟、让 TPOT/TTFT 抖动；PFC 兜底防丢，ECN/DCQCN 细粒度降速。")

    # ---- assertions: the ledger must be self-consistent ------------------ #
    assert a["drops"] == 0.0 and c["drops"] == 0.0, "PFC must prevent overflow drops"
    assert b["drops"] > 0.0, "without PFC, buffer overflow must show up"
    assert a["pause_events"] > 0 and c["pause_events"] * 5 < a["pause_events"], "ECN must cut PFC pauses"
    assert c["throughput"] > a["throughput"], "ECN+PFC should keep the link busier than PFC-only"
    assert c["avg_queue"] < a["avg_queue"], "ECN should keep the queue lower"
    assert c["mark_rate"] > 0.0 and a["mark_rate"] == 0.0
    assert b["mark_rate"] > c["mark_rate"], "no PFC -> higher standing queue -> more marks"
    assert simulate(SCENARIOS[2]) == c, "the simulation must be deterministic"
    for lo, hi in zip(scan_rows, scan_rows[1:]):
        assert lo["mark_rate"] > hi["mark_rate"], "higher ECN thresholds mark less"
        assert lo["pause_events"] < hi["pause_events"], "later ECN -> more PFC pauses"
    assert scan_rows[3]["mark_rate"] == 0.0, "max above XOFF means PFC fires first"
    assert R_MIN_MB <= c["final_rate"] <= R_MAX_MB
    assert 0.8 * LINK_MB <= c["throughput"] <= LINK_MB

    print(f"\nOK: PFC drops=0, ECN cuts pauses {a['pause_events']} -> {c['pause_events']}, "
          f"mark rate follows thresholds, DCQCN converges near line rate.")


if __name__ == "__main__":
    main()
