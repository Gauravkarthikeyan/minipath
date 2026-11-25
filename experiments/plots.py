"""Render every figure in docs/figures from results/*.json.

python -m experiments.plots
"""

from __future__ import annotations

import numpy as np

from experiments.common import FIGURES, load, style

plt = style()

# Validated categorical order (blue, orange, aqua); text stays neutral.
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
INK, INK2, GRID = "#191919", "#666666", "#d9d9d9"
LW = 2.0


def _finish(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def _label_end(ax, x, y, text, color=INK, dx=4, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", fontsize=9, color=color)


# ------------------------------------------------------------------ E1


def fig_dispatch():
    d = load("e1_dispatch")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    modes = [
        ("spmd", "SPMD (multi-controller)", BLUE),
        ("async", "single controller, async", ORANGE),
        ("sync", "single controller, sync", AQUA),
    ]
    rows = d["part_a"]
    xs = np.arange(len(rows))
    w = 0.26
    for i, (m, label, col) in enumerate(modes):
        vals = [r[m]["steps_per_sec"] for r in rows]
        bars = a.bar(xs + (i - 1) * w, vals, w - 0.03, color=col, label=label)
        for bar, v in zip(bars, vals):
            a.annotate(
                f"{v:,.0f}",
                (bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=INK2,
            )
    a.set_yscale("log")
    a.set_xticks(xs, [f"{r['workers']} worker{'s' if r['workers'] > 1 else ''}" for r in rows])
    a.set_ylabel("computations / second (log)")
    a.set_title("Trivial gang-scheduled computations")
    a.legend(loc="upper right", fontsize=8)

    rows = d["part_b"]
    fl = [r["flops"] for r in rows]
    for m, label, col in modes:
        ms = [r[m]["sec_per_step"] * 1e3 for r in rows]
        b.plot(fl, ms, "-o", color=col, lw=LW, ms=5, label=label)
    b.set_xscale("log")
    b.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, NullFormatter

    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.yaxis.set_minor_formatter(NullFormatter())
    b.set_yticks([1, 2, 5, 10, 20])
    b.set_xlabel("FLOPs per computation (matmul n×n, n = 16 … 1024)")
    b.set_ylabel("ms per computation (log)")
    b.set_title("Dispatch cost vs. compute, 3 workers")
    b.legend(fontsize=8)
    _finish(fig, "e1_dispatch")


# ------------------------------------------------------------------ E2


def fig_gang():
    d = load("e2_gang")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    names = [
        ("unscheduled_random", "no scheduler,\nrandom send order"),
        ("unscheduled_fixed", "no scheduler,\nfixed send order"),
        ("gang_scheduled", "central gang\nscheduler"),
    ]
    rates = [d["runs"][k]["deadlock_rate"] * 100 for k, _ in names]
    cols = [INK2, INK2, ORANGE]
    bars = a.barh([n for _, n in names], rates, color=cols, height=0.55)
    for bar, r, (k, _) in zip(bars, rates, names):
        dl, n = d["runs"][k]["deadlocks"], d["n_trials"]
        a.annotate(
            f"{dl}/{n} trials deadlocked",
            (r, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    a.set_xlim(0, 135)
    a.set_xticks([0, 25, 50, 75, 100])
    a.invert_yaxis()
    a.set_xlabel("% of trials that deadlocked")
    a.set_title(f"2 clients × {d['ops_per_client']} all-reduces on 4 shared workers")
    a.grid(axis="y", visible=False)

    sw = d["sweep_unscheduled_fixed"]
    ks = [r["ops_per_client"] for r in sw]
    b.plot(ks, [r["deadlock_rate"] * 100 for r in sw], "-o", color=INK2, lw=LW, ms=6)
    b.set_xscale("log", base=2)
    b.set_xticks(ks, [str(k) for k in ks])
    b.set_ylim(-5, 105)
    b.set_xlabel("all-reduces per client")
    b.set_ylabel("% trials deadlocked")
    b.set_title("Without a scheduler, it's a matter of time")
    _finish(fig, "e2_gang")


# ------------------------------------------------------------------ E3


def _gantt(ax, trace: dict, rows: list, labels: list, colors: dict, t_lo: float, t_hi: float):
    for i, w in enumerate(rows):
        for name, cat, t0, t1 in trace[str(w)]:
            if t1 < t_lo or t0 > t_hi:
                continue
            col = colors.get(name, colors.get(cat, INK2))
            ax.barh(
                i, (t1 - t0) * 1e3, left=(t0 - t_lo) * 1e3, height=0.62, color=col, edgecolor="white", linewidth=0.6
            )
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, (t_hi - t_lo) * 1e3)
    ax.grid(axis="y", visible=False)


def _one_step_window(trace: dict, worker: str, n_steps_back: int = 1):
    """Window covering the last full step: from the first fwd after the
    second-last adam on ``worker`` to its last adam."""
    adams = [e for e in trace[worker] if e[0] == "adam"]
    lo = adams[-1 - n_steps_back][3]
    hi = max(e[3] for evs in trace.values() for e in evs if e[0] == "adam")
    return lo, hi


def fig_pipeline():
    d = load("e3_pipeline")
    row = next(r for r in d["sweep"] if r["M"] == 4)
    tr = row["trace"]
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.55, wspace=0.28)
    a = fig.add_subplot(gs[0, :])
    workers = sorted(tr, key=int)
    lo, hi = _one_step_window(tr, workers[0])
    colors = {
        "fwd": BLUE,
        "bwd": ORANGE,
        "fwd+bwd": ORANGE,
        "accum": AQUA,
        "adam": AQUA,
        "norm": AQUA,
        "batch": "#b3b3b3",
        "collective": AQUA,
    }
    _gantt(a, tr, [int(w) for w in workers], [f"stage {i}" for i in range(len(workers))], colors, lo, hi)
    a.set_xlabel("ms (one training step, measured on the workers)")
    a.set_title("GPipe schedule emerging from in-order streams + futures (4 stages, 4 micro-batches)")
    from matplotlib.patches import Patch

    a.legend(
        handles=[
            Patch(color=BLUE, label="forward"),
            Patch(color=ORANGE, label="backward"),
            Patch(color=AQUA, label="grad accumulate / norm / Adam"),
            Patch(color="#b3b3b3", label="batch"),
        ],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        fontsize=8,
    )

    b = fig.add_subplot(gs[1, 0])
    Ms = [r["M"] for r in d["sweep"]]
    b.plot(
        Ms, [r["bubble_theory"] * 100 for r in d["sweep"]], "--", color=INK2, lw=LW, label="GPipe bound (P−1)/(M+P−1)"
    )
    b.plot(
        Ms,
        [r["bubble_measured"] * 100 for r in d["sweep"]],
        "-o",
        color=ORANGE,
        lw=LW,
        ms=6,
        label="measured idle time",
    )
    b.set_xscale("log", base=2)
    b.set_xticks(Ms, [str(m) for m in Ms])
    b.set_xlabel("micro-batches M")
    b.set_ylabel("pipeline bubble (%)")
    b.set_title("Bubble: measured vs. theory")
    b.legend(fontsize=8)

    c = fig.add_subplot(gs[1, 1])
    eq = d["equivalence"]
    steps = np.arange(len(eq["single"]))
    c.plot(steps, eq["single"], color=BLUE, lw=4, alpha=0.35, label="1 device")
    c.plot(steps, eq["pipeline"], color=ORANGE, lw=1.4, label="4-stage pipeline")
    c.set_xlabel("step")
    c.set_ylabel("train loss (nats/char)")
    c.set_title(f"Same math: max |Δloss| = {eq['max_abs_diff']:.1e}")
    c.legend(fontsize=8)
    _finish(fig, "e3_pipeline")


# ------------------------------------------------------------------ E4


def fig_fault():
    d = load("e4_fault_tolerance")
    runs = d["runs"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    series = [
        ("clean_spmd", "no failure", "#b3b3b3"),
        ("spmd", "SPMD + checkpoint/restart", BLUE),
        ("single_controller", "single controller + spare / elastic", ORANGE),
    ]
    for key, label, col in series:
        r = runs[key]
        t = np.array([x["t"] - r["t0"] for x in r["log"]])
        s = np.array([x["step"] for x in r["log"]])
        a.plot(t, s, color=col, lw=LW, label=label)
        _label_end(a, t[-1], s[-1], f"{t[-1]:.0f}s", color=INK)
    for key, col in (("spmd", BLUE), ("single_controller", ORANGE)):
        r = runs[key]
        for e in r["events"]:
            if e["kind"] == "kill":
                a.axvline(e["t"] - r["t0"], color=col, lw=1, ls=":")
    k1 = d["kills"]["first_after_step"]
    a.annotate(
        f"worker killed after step {k1}",
        (next(e["t"] - runs["spmd"]["t0"] for e in runs["spmd"]["events"] if e["kind"] == "kill"), k1),
        xytext=(-8, 40),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK2,
        arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8},
    )
    a.set_xlabel("wall-clock seconds")
    a.set_ylabel("training step completed")
    a.set_title("Progress under failures")
    a.legend(fontsize=8, loc="upper left")

    for key, label, col, lw in (
        ("clean", "no failure", "#b3b3b3", 4),
        ("single_controller", "recovered (spare, then elastic 4→3)", ORANGE, 1.3),
    ):
        r = runs[key]
        last = {}
        for x in r["log"]:
            last[x["step"]] = x["loss"]
        ks = sorted(last)
        b.plot(ks, [last[k] for k in ks], color=col, lw=lw, alpha=0.5 if lw > 2 else 1, label=label)
    k2 = d["kills"]["second_after_step"]
    b.axvline(k1, color=INK2, lw=1, ls=":")
    b.axvline(k2, color=INK2, lw=1, ls=":")
    b.text(k1, b.get_ylim()[1], " kill → spare", fontsize=8, color=INK2, va="top")
    b.text(k2, b.get_ylim()[1], " kill → shrink", fontsize=8, color=INK2, va="top")
    b.set_xlabel("step")
    b.set_ylabel("train loss (nats/char)")
    b.set_title("Bit-identical until the elastic shrink" if d["bit_identical_until_shrink"] else "Loss after recovery")
    b.legend(fontsize=8, loc="center right")
    _finish(fig, "e4_fault_tolerance")


# ------------------------------------------------------------------ E5


def fig_rl():
    d = load("e5_rl_overlap")
    runs = d["runs"]
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.6, wspace=0.28)
    colors = {"rollout": BLUE, "learn": ORANGE, "adam": AQUA, "collective": AQUA}
    span = 0.0
    traced = [r for r in runs if "trace" in r]
    for r in traced:
        span = max(span, r["trace_span"])
    for col_i, r in enumerate(sorted(traced, key=lambda r: r["mode"] != "sync")):
        ax = fig.add_subplot(gs[0, col_i])
        rows = r["sampler_workers"] + r["learner_workers"]
        _gantt(ax, r["trace"], rows, ["sampler 0", "sampler 1", "learner 0", "learner 1"], colors, 0, span)
        u = r["util"]
        ax.set_title(f"{r['mode']}: sampler busy {u['sampler']:.0%}, learner busy {u['learner']:.0%}")
        ax.set_xlabel("ms (5 iterations)")
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(color=BLUE, label="rollout (sampler program)"),
            Patch(color=ORANGE, label="policy gradient (learner program)"),
            Patch(color=AQUA, label="all-reduce + Adam"),
        ],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.515),
        fontsize=8,
    )

    a = fig.add_subplot(gs[1, 0])
    b = fig.add_subplot(gs[1, 1])
    for mode, col in (("sync", BLUE), ("async", ORANGE)):
        rs = [r for r in runs if r["mode"] == mode]
        n = min(len(r["log"]) for r in rs)
        it = np.array([r["log"][i]["iter"] for i in range(n) for r in rs[:1]])
        ret = np.array([[r["log"][i]["return"] for i in range(n)] for r in rs])
        tt = np.array([[r["log"][i]["t"] for i in range(n)] for r in rs]).mean(0)
        mu, lo, hi = ret.mean(0), ret.min(0), ret.max(0)
        a.plot(it, mu, color=col, lw=LW, label=mode)
        a.fill_between(it, lo, hi, color=col, alpha=0.15, lw=0)
        b.plot(tt, mu, color=col, lw=LW, label=mode)
        b.fill_between(tt, lo, hi, color=col, alpha=0.15, lw=0)
    for ax, xl in ((a, "iteration"), (b, "wall-clock seconds")):
        ax.axhline(d["threshold"], color=INK2, lw=1, ls=":")
        ax.set_xlabel(xl)
        ax.set_ylabel("mean episode return")
        ax.legend(fontsize=8, loc="lower right")
    a.set_title("Sample efficiency (lag-1 costs little)")
    b.set_title("Time to solve CartPole")
    _finish(fig, "e5_rl_overlap")


def main():
    for f in (fig_dispatch, fig_gang, fig_pipeline, fig_fault, fig_rl):
        try:
            f()
        except FileNotFoundError as e:
            print(f"skip {f.__name__}: {e}")


if __name__ == "__main__":
    main()
