#!/usr/bin/env python3
"""Plot the results of quadrotor_manifold_bench.

Usage: python3 plot_quadrotor_manifold_bench.py RESULT_DIR

Reads ocp_results.csv, closed_loop_summary.csv and closed_loop_steps.csv from
RESULT_DIR and writes PNG figures plus summary.md into the same directory.
Only numpy and matplotlib are required.
"""
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FORMULATIONS = ["manifold", "manifold_proj", "quat_norm", "quat_free", "euler"]
FORMULATION_LABELS = {
    "manifold": "manifold\n(Lie RK4)",
    "manifold_proj": "manifold\n(proj. RK4)",
    "quat_norm": "quat + |q|=1",
    "quat_free": "quat (free)",
    "euler": "Euler ZYX",
}
SOLVERS = ["fatrop", "fatrop_sd", "fatrop_jit", "fatrop_jit_sd", "ipopt", "ipopt_sd",
           "ipopt_jit", "ipopt_jit_sd", "sqpmethod"]
SOLVER_COLORS = {"fatrop": "#1f77b4", "fatrop_sd": "#9467bd", "fatrop_jit": "#17becf",
                 "fatrop_jit_sd": "#d62728", "ipopt": "#ff7f0e", "ipopt_sd": "#8c564b",
                 "ipopt_jit": "#bcbd22", "ipopt_jit_sd": "#e377c2", "sqpmethod": "#2ca02c"}


SCENARIO = ""  # name and description from scenario.txt (shown as figure title)


def save(fig, path):
    if SCENARIO:
        fig.suptitle(SCENARIO, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def relative_gap(rows, key, out_key):
    """(J - J_best) / J_best per scenario, J_best among successful runs"""
    best = {}
    for r in rows:
        c = num(r.get(key, "nan"))
        if r["success"] == "1" and math.isfinite(c):
            best[r["scenario"]] = min(best.get(r["scenario"], math.inf), c)
    for r in rows:
        c = num(r.get(key, "nan"))
        b = best.get(r["scenario"], math.nan)
        r[out_key] = str((c - b) / b) if math.isfinite(c) and math.isfinite(b) and b > 0 else "nan"


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def combos(rows):
    present = {(r["formulation"], r["solver"]) for r in rows}
    return [(f, s) for f in FORMULATIONS for s in SOLVERS if (f, s) in present]


def select(rows, f, s):
    return [r for r in rows if r["formulation"] == f and r["solver"] == s]


def grouped_positions(n_groups, n_in_group, width=0.8):
    step = width / n_in_group
    return [[g - width / 2 + step * (i + 0.5) for i in range(n_in_group)] for g in range(n_groups)], step


def bar_by_combo(ax, rows, value, ylabel, title, ylim=None, fmt="{:.0%}"):
    pos, step = grouped_positions(len(FORMULATIONS), len(SOLVERS))
    for si, s in enumerate(SOLVERS):
        xs, ys = [], []
        for fi, f in enumerate(FORMULATIONS):
            sel = select(rows, f, s)
            if not sel:
                continue
            xs.append(pos[fi][si])
            ys.append(value(sel))
        if xs:
            bars = ax.bar(xs, ys, step * 0.9, color=SOLVER_COLORS[s], label=s)
            for b, y in zip(bars, ys):
                if not math.isnan(y):
                    ax.annotate(fmt.format(y), (b.get_x() + b.get_width() / 2, y),
                                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(FORMULATIONS)))
    ax.set_xticklabels([FORMULATION_LABELS[f] for f in FORMULATIONS])
    ax.set_xlim(-0.5, len(FORMULATIONS) - 0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def box_by_combo(ax, rows, key, ylabel, title, log=False, only_success=False, success_key="success",
                 floor=None):
    """Box plot per (formulation, solver); on a log scale values below `floor` are shown at it."""
    pos, step = grouped_positions(len(FORMULATIONS), len(SOLVERS))
    for si, s in enumerate(SOLVERS):
        data, xs = [], []
        for fi, f in enumerate(FORMULATIONS):
            sel = select(rows, f, s)
            if only_success:
                sel = [r for r in sel if r[success_key] == "1"]
            vals = [num(r[key]) for r in sel]
            if floor is not None:
                vals = [max(v, floor) for v in vals if math.isfinite(v)]
            vals = [v for v in vals if math.isfinite(v) and (v > 0 or not log)]
            if not vals:
                continue
            data.append(vals)
            xs.append(pos[fi][si])
        if data:
            bp = ax.boxplot(data, positions=xs, widths=step * 0.8, patch_artist=True,
                            showfliers=True, manage_ticks=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(SOLVER_COLORS[s])
                patch.set_alpha(0.6)
            ax.plot([], [], color=SOLVER_COLORS[s], linewidth=8, alpha=0.6, label=s)
    ax.set_xticks(range(len(FORMULATIONS)))
    ax.set_xticklabels([FORMULATION_LABELS[f] for f in FORMULATIONS])
    ax.set_xlim(-0.5, len(FORMULATIONS) - 0.5)
    if floor is not None:
        ax.axhline(floor, color="gray", linewidth=0.8, linestyle=":")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, which="both")


def success_rate(sel):
    return sum(r["success"] == "1" for r in sel) / len(sel) if sel else math.nan


def plot_ocp(rows, out):
    if not rows:
        return
    # relative true cost w.r.t. the best solution of each scenario (among successful solves)
    relative_gap(rows, "true_cost", "cost_gap")

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    bar_by_combo(axs[0, 0], rows, success_rate, "success rate",
                 "OCP (cold start): solver success rate", ylim=(0, 1.1))
    box_by_combo(axs[0, 1], rows, "solve_time_ms", "solve time [ms]",
                 "OCP: solve time (all solves)", log=True)
    box_by_combo(axs[1, 0], rows, "iterations", "iterations",
                 "OCP: iterations (all solves)", log=True)
    box_by_combo(axs[1, 1], rows, "cost_gap", "(J - J_best) / J_best",
                 "OCP: true cost of the plant rollout vs best (all solves, floor 1e-8)",
                 log=True, floor=1e-8)
    save(fig, os.path.join(out, "ocp_overview.png"))
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    box_by_combo(axs[0], rows, "pred_att_err_deg", "max attitude error [deg]",
                 "OCP: prediction vs plant (attitude)", log=True, floor=1e-6)
    box_by_combo(axs[1], rows, "terminal_att_err_deg", "attitude error [deg]",
                 "OCP: plant attitude error at the horizon end", log=True)
    box_by_combo(axs[2], rows, "quat_norm_dev", "max | |q| - 1 |",
                 "OCP: quaternion norm drift of the prediction (floor 1e-16, Euler: n/a)",
                 log=True, floor=1e-16)
    save(fig, os.path.join(out, "ocp_accuracy.png"))
    plt.close(fig)

    # success rate versus initial attitude angle
    fig, axs = plt.subplots(1, len(SOLVERS), figsize=(6 * len(SOLVERS), 4.5), sharey=True)
    bins = np.linspace(0, 180, 7)
    for ax, s in zip(np.atleast_1d(axs), SOLVERS):
        for f in FORMULATIONS:
            sel = select(rows, f, s)
            if not sel:
                continue
            ang = np.array([num(r["angle_deg"]) for r in sel])
            ok = np.array([r["success"] == "1" for r in sel], dtype=float)
            centers, rates = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (ang >= lo) & (ang < hi)
                if m.any():
                    centers.append((lo + hi) / 2)
                    rates.append(ok[m].mean())
            ax.plot(centers, rates, marker="o", label=FORMULATION_LABELS[f])
        ax.set_title(f"{s}: success rate vs initial tilt")
        ax.set_xlabel("initial attitude error [deg]")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    np.atleast_1d(axs)[0].set_ylabel("success rate")
    save(fig, os.path.join(out, "ocp_success_vs_angle.png"))
    plt.close(fig)


def plot_closed_loop(summary, steps, out):
    if not summary:
        return
    relative_gap(summary, "closed_loop_cost", "cl_cost_gap")
    fig, axs = plt.subplots(2, 3, figsize=(21, 10))
    bar_by_combo(axs[0, 0], summary, success_rate, "success rate",
                 "Closed loop: reached the goal (|p| < 0.1 m, attitude < 5 deg)", ylim=(0, 1.1))
    box_by_combo(axs[0, 2], summary, "cl_cost_gap", "(J_cl - J_best) / J_best",
                 "Closed loop: accumulated plant cost vs best (floor 1e-6)", log=True, floor=1e-6)
    box_by_combo(axs[1, 2], summary, "final_pos_err", "position error [m]",
                 "Closed loop: final position error", log=True)
    box_by_combo(axs[0, 1], steps, "solve_time_ms", "solve time [ms]",
                 "Closed loop: solve time per MPC step (warm start)", log=True)
    bar_by_combo(axs[1, 0], steps,
                 lambda sel: 1 - sum(r["success"] == "1" for r in sel) / len(sel),
                 "fraction of failed solves", "Closed loop: failed solver calls",
                 fmt="{:.1%}")
    box_by_combo(axs[1, 1], summary, "final_att_err_deg", "attitude error [deg]",
                 "Closed loop: final attitude error", log=True)
    save(fig, os.path.join(out, "closed_loop_overview.png"))
    plt.close(fig)

    # attitude error over time for the hardest scenario
    angles = {r["scenario"]: num(r["angle_deg"]) for r in summary}
    hardest = max(angles, key=angles.get)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    for f in FORMULATIONS:
        for s in SOLVERS:
            sel = [r for r in steps if r["formulation"] == f and r["solver"] == s
                   and r["scenario"] == hardest]
            if not sel:
                continue
            t = [num(r["time"]) for r in sel]
            ls = {"fatrop": "-", "fatrop_sd": (0, (3, 1)), "fatrop_jit": "-.", "fatrop_jit_sd": "-",
          "ipopt": "--", "sqpmethod": ":"}.get(s, "-")
            axs[0].plot(t, [num(r["att_err_deg"]) for r in sel], linestyle=ls, label=f"{f}/{s}")
            axs[1].plot(t, [num(r["pos_err"]) for r in sel], linestyle=ls, label=f"{f}/{s}")
    axs[0].set_title(f"Closed loop, initial tilt {angles[hardest]:.0f} deg: attitude error")
    axs[0].set_ylabel("attitude error [deg]")
    axs[1].set_title("position error")
    axs[1].set_ylabel("|p| [m]")
    for ax in axs:
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    save(fig, os.path.join(out, "closed_loop_trajectory.png"))
    plt.close(fig)


def median(vals):
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.median(vals)) if vals else math.nan


def write_summary(ocp, summary, steps, out, builds=()):
    lines = ["# Quadrotor manifold benchmark\n"]
    if SCENARIO:
        lines += [f"Scenario: {SCENARIO}\n"]
    if builds:
        lines += ["## Build (construction incl. JIT compilation)\n",
                  "| formulation | solver | build time [s] |", "|---|---|---|"]
        for r in builds:
            lines.append(f"| {r['formulation']} | {r['solver']} | "
                         f"{num(r['build_time_ms']) / 1e3:.1f} |")
        lines.append("")
    if ocp:
        lines += ["## OCP (cold start)\n",
                  "| formulation | solver | n | success | median time [ms] | median iters | "
                  "median cost gap | median |q|-1 drift |",
                  "|---|---|---|---|---|---|---|---|"]
        for f, s in combos(ocp):
            sel = select(ocp, f, s)
            lines.append(
                f"| {f} | {s} | {len(sel)} | {success_rate(sel):.0%} | "
                f"{median([num(r['solve_time_ms']) for r in sel]):.1f} | "
                f"{median([num(r['iterations']) for r in sel]):.0f} | "
                f"{median([num(r.get('cost_gap', 'nan')) for r in sel]):.2e} | "
                f"{median([num(r['quat_norm_dev']) for r in sel]):.1e} |")
        lines.append("")
    if summary:
        lines += ["## Closed loop\n",
                  "| formulation | solver | n | success | failed solves | median time [ms] | "
                  "max time [ms] | median final attitude error [deg] | median closed-loop cost gap |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for f, s in combos(summary):
            sel = select(summary, f, s)
            st = select(steps, f, s)
            failed = 1 - sum(r["success"] == "1" for r in st) / len(st) if st else math.nan
            lines.append(
                f"| {f} | {s} | {len(sel)} | {success_rate(sel):.0%} | {failed:.1%} | "
                f"{median([num(r['solve_time_ms']) for r in st]):.1f} | "
                f"{max([num(r['max_solve_ms']) for r in sel]):.1f} | "
                f"{median([num(r['final_att_err_deg']) for r in sel]):.2f} | "
                f"{median([num(r.get('cl_cost_gap', 'nan')) for r in sel]):.2e} |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def restrict_solvers(*tables):
    """Only lay out the solvers present in the results"""
    global SOLVERS
    present = {r["solver"] for t in tables for r in t}
    SOLVERS = [s for s in SOLVERS if s in present] or SOLVERS


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    out = sys.argv[1]
    global SCENARIO
    meta = os.path.join(out, "scenario.txt")
    if os.path.exists(meta):
        with open(meta) as f:
            SCENARIO = " - ".join(l.strip() for l in f if l.strip())
    ocp = read_csv(os.path.join(out, "ocp_results.csv"))
    summary = read_csv(os.path.join(out, "closed_loop_summary.csv"))
    steps = read_csv(os.path.join(out, "closed_loop_steps.csv"))
    builds = read_csv(os.path.join(out, "build_times.csv"))
    restrict_solvers(ocp, summary)
    plot_ocp(ocp, out)
    plot_closed_loop(summary, steps, out)
    write_summary(ocp, summary, steps, out, builds)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
