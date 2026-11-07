#!/usr/bin/env python3
"""
figures.py — Generate cross-model figures.

Reads every results/<model>/eval_summary.json and produces:

  figures/fig4_effects.pdf        — Paired Δ refusal rate vs baseline, per
                                    primitive × attack cell × model, with
                                    bootstrap 95% CIs.
  figures/fig5_utility.pdf        — Safety–utility Pareto: arrow per model
                                    from baseline to both.

Figures are model-agnostic: adding an N-th model auto-appears.

Usage
-----
  python figures.py
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch
from scipy import stats


RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR     = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Display configuration
# ---------------------------------------------------------------------------
CONDITIONS        = ["baseline", "tags", "capabilities", "both"]
CONDS_VS_BASELINE = ["tags", "capabilities", "both"]
CONDS_MAIN        = ["tags", "capabilities", "both"]
COND_LABEL = {  # display names used in the paper
    "baseline":     "none",
    "tags":         "source",
    "capabilities": "scopes",
    "both":         "full",
}
COND_LABEL_FLAT = {k: v.replace("\n", " ") for k, v in COND_LABEL.items()}

VARIANTS = ["trusted", "untrusted", "in_scope", "out_of_scope"]
VARIANT_TITLE = {
    "trusted":      ("Legitimate request", "(agent should act)"),
    "untrusted":    ("Unauthorized",       "(agent should refuse)"),
    "in_scope":     ("In-scope action",    "(agent should act)"),
    "out_of_scope": ("Out-of-scope action","(agent should refuse)"),
}

TYPES = ["source_label_ipi", "scope_check_ipi", "confused_deputy"]
TYPE_TITLE = {
    "source_label_ipi": ("Source-label IPI",  ""),
    "scope_check_ipi":  ("Scope-check IPI",   ""),
    "confused_deputy":  ("Confused deputy",   ""),
}

MODEL_LABEL = {
    "claude-sonnet-4-6":                       "Sonnet 4.6",
    "openrouter-deepseek-deepseek-v3.2":       "DeepSeek V3.2",
    "openrouter-openai-gpt-5.4":               "GPT-5.4",
    "gemini-gemini-3.1-pro-preview":           "Gemini 3.1 Pro",
    "openrouter-moonshotai-kimi-k2.6":         "Kimi K2.6",
    "openrouter-minimax-minimax-m2.7":         "MiniMax M2.7",
    "openrouter-z-ai-glm-5.1":                 "GLM 5.1",
    "openrouter-qwen-qwen3.6-plus":            "Qwen 3.6 Plus",
    "openrouter-x-ai-grok-4.20":               "Grok 4.20",
    "openrouter-xiaomi-mimo-v2.5-pro":         "MiMo v2.5 Pro",
}
SOURCE_TYPE = {
    "claude-sonnet-4-6":                       "closed",
    "openrouter-openai-gpt-5.4":               "closed",
    "gemini-gemini-3.1-pro-preview":           "closed",
    "openrouter-qwen-qwen3.6-plus":            "closed",
    "openrouter-x-ai-grok-4.20":               "closed",
    "openrouter-deepseek-deepseek-v3.2":       "open",
    "openrouter-moonshotai-kimi-k2.6":         "open",
    "openrouter-minimax-minimax-m2.7":         "open",
    "openrouter-xiaomi-mimo-v2.5-pro":         "open",
    "openrouter-z-ai-glm-5.1":                 "open",
}
# Display ordering: closed-source first, then open-source (within each group,
# alphabetical by display name to keep the ordering stable across re-runs).
MODEL_ORDER = [
    "claude-sonnet-4-6", "gemini-gemini-3.1-pro-preview",
    "openrouter-openai-gpt-5.4", "openrouter-qwen-qwen3.6-plus",
    "openrouter-x-ai-grok-4.20",
    "openrouter-deepseek-deepseek-v3.2", "openrouter-z-ai-glm-5.1",
    "openrouter-moonshotai-kimi-k2.6", "openrouter-xiaomi-mimo-v2.5-pro",
    "openrouter-minimax-minimax-m2.7",
]
MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

def load_model(results_dir: Path) -> dict | None:
    # Read raw records — eval_summary.json only retains the most recent run's
    # filtered subset, so we ingest the streamed jsonl which holds the full
    # cross-condition history (last write wins per (scenario, condition)).
    rp = results_dir / "eval_records.jsonl"
    if not rp.exists():
        return None
    records: dict[tuple, dict] = {}
    for line in rp.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        records[(r["scenario"], r.get("condition", "baseline"))] = r
    c, p = Counter(), Counter()
    per_scenario: dict = {}
    for r in records.values():
        k = (r.get("condition"), r["type"], r["variant"])
        c[k] += 1
        if r["passed"]: p[k] += 1
        scen = r["scenario"]
        per_scenario.setdefault((r["type"], r["variant"], scen), {})[r["condition"]] = r["passed"]
    cells = {k: {"n": c[k], "pass": p[k],
                 "pass_rate": p[k]/c[k] if c[k] else 0}
             for k in c}
    return {
        "slug":         results_dir.name,
        "display":      MODEL_LABEL.get(results_dir.name, results_dir.name),
        "cells":        cells,
        "per_scen":     per_scenario,
        "raw_records":  list(records.values()),
    }


def discover_models() -> list[dict]:
    models = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir(): continue
        m = load_model(d)
        if m is not None: models.append(m)
    # Reorder: closed-source first, then open-source (per MODEL_ORDER).
    # Anything missing from MODEL_ORDER falls to the end in original order.
    rank = {slug: i for i, slug in enumerate(MODEL_ORDER)}
    models.sort(key=lambda m: rank.get(m["slug"], len(MODEL_ORDER)))
    return models


# ---------------------------------------------------------------------------
# Paired-effect computation
# ---------------------------------------------------------------------------

def paired_effect(model, type_, variant, cond_a, cond_b, n_boot=2000):
    """
    Return Δ pass rate (cond_b − cond_a) with bootstrap 95% CI, exact McNemar
    p, and discordance counts for the specified cell. `pass` = correct outcome
    on that variant (called on trusted, blocked on attack variants).
    """
    pa_list, pb_list = [], []
    for (t, v, _), passes in model["per_scen"].items():
        if t != type_ or v != variant: continue
        a, b = passes.get(cond_a), passes.get(cond_b)
        if a is None or b is None: continue
        pa_list.append(int(a)); pb_list.append(int(b))
    n = len(pa_list)
    if n == 0: return None

    pa = np.array(pa_list); pb = np.array(pb_list)
    delta = (pb.mean() - pa.mean()) * 100

    # Bootstrap CI
    seed = abs(hash((model["slug"], type_, variant, cond_a, cond_b))) % (2**32)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = (pb[idx].mean(axis=1) - pa[idx].mean(axis=1)) * 100
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])

    # McNemar: n10 = A fail, B pass (improvement); n01 = A pass, B fail (regression)
    n10 = int(((pa == 0) & (pb == 1)).sum())
    n01 = int(((pa == 1) & (pb == 0)).sum())
    n_d = n10 + n01
    p_val = (stats.binomtest(min(n10, n01), n_d, 0.5, alternative="two-sided").pvalue
             if n_d else None)
    return dict(delta=delta, ci_lo=ci_lo, ci_hi=ci_hi,
                n10=n10, n01=n01, n_d=n_d, p=p_val, n=n)


def sig_marker(p):
    if p is None: return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def sig_hatch(p):
    """Bar-fill hatch indicating significance vs baseline. None = no hatch."""
    if p is None: return None
    if p < 0.05:  return "////"
    return None


SIG_NOTE = ("Hatched bars: change vs no-primitives baseline is significant "
            "(paired McNemar's exact test, p<0.05).")


# ---------------------------------------------------------------------------
# Joint-pass-per-pair (trusted-correct AND untrusted-correct) helpers
# ---------------------------------------------------------------------------

def _pair_id(scenario_id):
    for suf in ("_transitive_2hop_trusted", "_transitive_2hop",
                "_out_of_scope", "_in_scope",
                "_untrusted", "_trusted"):
        if scenario_id.endswith(suf):
            try:
                return int(scenario_id[:-len(suf)].rsplit("_", 1)[1])
            except ValueError:
                return None
    return None


def joint_pass_vector(model, type_, var_t, var_u, cond):
    """
    Return ([pair_ids], [0/1]) — for each pair_id where both trusted and
    untrusted variants are observed under `cond`, the joint indicator
    (trusted_passed AND untrusted_passed).
    """
    by_pair: dict[int, dict[str, bool]] = {}
    for r in model.get("raw_records") or []:
        if r["type"] != type_:                     continue
        if r.get("condition") != cond:             continue
        if r["variant"] not in (var_t, var_u):     continue
        pid = _pair_id(r["scenario"])
        if pid is None: continue
        by_pair.setdefault(pid, {})[r["variant"]] = bool(r["passed"])
    pids = sorted(p for p, d in by_pair.items() if var_t in d and var_u in d)
    joint = np.array([int(by_pair[p][var_t] and by_pair[p][var_u]) for p in pids])
    return pids, joint


def joint_paired_effect(model, type_, var_t, var_u, cond_a, cond_b):
    """
    McNemar paired test on the joint-pass indicator between two conditions
    (per scenario pair). Returns rate_a, rate_b, delta, n10/n01/n_d, p, n.
    """
    pids_a, ja = joint_pass_vector(model, type_, var_t, var_u, cond_a)
    pids_b, jb = joint_pass_vector(model, type_, var_t, var_u, cond_b)
    common = sorted(set(pids_a) & set(pids_b))
    if not common: return None
    ia = {p: i for i, p in enumerate(pids_a)}
    ib = {p: i for i, p in enumerate(pids_b)}
    pa = np.array([ja[ia[p]] for p in common])
    pb = np.array([jb[ib[p]] for p in common])
    n10 = int(((pa == 0) & (pb == 1)).sum())
    n01 = int(((pa == 1) & (pb == 0)).sum())
    n_d = n10 + n01
    p_val = (stats.binomtest(min(n10, n01), n_d, 0.5, alternative="two-sided").pvalue
             if n_d else None)
    return dict(rate_a=pa.mean() * 100, rate_b=pb.mean() * 100,
                delta=(pb.mean() - pa.mean()) * 100,
                n10=n10, n01=n01, n_d=n_d, p=p_val, n=len(common))


VARIANT_PAIR = {
    "source_label_ipi":     ("trusted",  "untrusted"),
    "scope_check_ipi": ("in_scope", "out_of_scope"),
    "confused_deputy":      ("trusted",  "untrusted"),
}


# ---------------------------------------------------------------------------
# Fig 4 — unified dumbbell: per-model joint pass rate before/after exposing
# the relevant primitive(s) on each of three attack types.
# ---------------------------------------------------------------------------

def fig1_headroom(models):
    """
    Headline figure for the dataset paper. Per model, two bars: joint pass
    rate at baseline (no primitives) and at the best primitive condition,
    averaged across the three attack families (source-label IPI, confused
    deputy, scope-check IPI). The "best" condition is selected per
    family as the max over {tags, capabilities, both}. Single y-axis from
    0 to 100 so the gap to perfect is visible directly.
    """
    fig, ax = plt.subplots(figsize=(10.0, 5.4))

    ATTACK_SPEC = [
        ("source_label_ipi",     "trusted",  "untrusted"),
        ("confused_deputy",      "trusted",  "untrusted"),
        ("scope_check_ipi", "in_scope", "out_of_scope"),
    ]

    BASELINE_COLOR = "#9ca3af"
    BEST_COLOR     = "#2b6cb0"
    bar_w = 0.38
    n_m = len(models)
    x_positions = np.arange(n_m)

    for mi, m in enumerate(models):
        base_rates, best_rates = [], []
        for atype, var_t, var_u in ATTACK_SPEC:
            base_r, best_r = None, 0.0
            for cond in CONDITIONS:
                _, j = joint_pass_vector(m, atype, var_t, var_u, cond)
                if not len(j): continue
                r = j.mean() * 100
                if cond == "baseline": base_r = r
                if r > best_r: best_r = r
            if base_r is None: continue
            base_rates.append(base_r); best_rates.append(best_r)
        if not base_rates: continue
        avg_base = float(np.mean(base_rates))
        avg_best = float(np.mean(best_rates))

        ax.bar(mi - bar_w/2, avg_base, bar_w, color=BASELINE_COLOR,
               label="no primitives (deployed default)" if mi == 0 else None,
               edgecolor="black", linewidth=0.4, zorder=2)
        ax.bar(mi + bar_w/2, avg_best, bar_w, color=BEST_COLOR,
               label="best primitive condition" if mi == 0 else None,
               edgecolor="black", linewidth=0.4, zorder=2)

    # 100% reference line.
    ax.axhline(100, color="#444", lw=0.7, ls="--", alpha=0.7, zorder=1)
    ax.text(n_m - 0.3, 100.8, "perfect (100%)",
            fontsize=9, color="#666", ha="right", va="bottom", style="italic")

    closed_idxs = [i for i, m in enumerate(models)
                   if SOURCE_TYPE.get(m["slug"]) == "closed"]
    open_idxs = [i for i, m in enumerate(models)
                 if SOURCE_TYPE.get(m["slug"]) == "open"]
    div_x = ((closed_idxs[-1] + open_idxs[0]) / 2
             if closed_idxs and open_idxs else None)
    if div_x is not None:
        ax.axvline(div_x, color="#cccccc", lw=0.7, ls=":", alpha=0.8, zorder=1)
        # Closed/open labels live in the empty band just below the 100% line so
        # they never overlap a bar (no model exceeds ~70 averaged).
        ax.text(div_x - 0.15, 88, "closed source",
                ha="right", va="center", fontsize=10, color="#555", style="italic")
        ax.text(div_x + 0.15, 88, "open source",
                ha="left", va="center", fontsize=10, color="#555", style="italic")

    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 10))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([m["display"] for m in models],
                       rotation=18, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlim(-0.7, n_m - 0.3)
    ax.set_ylabel("Joint pass rate (%)\n(averaged across families)",
                  fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Legend sits in the upper-left empty band — no bar gets close to it.
    ax.legend(loc="upper left", fontsize=10.5,
              bbox_to_anchor=(0.01, 0.99),
              frameon=True, edgecolor="#bbb", framealpha=0.95)

    fig.subplots_adjust(bottom=0.18, top=0.96, left=0.09, right=0.98)

    out = FIG_DIR / "fig1_headroom.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Fig 4 (legacy) — paired Δ-from-baseline effects with bootstrap CIs + MDE band
# Now demoted to an appendix figure: per-primitive decomposition (4 conditions
# per model) showing which primitive drives each 1-hop panel.
# ---------------------------------------------------------------------------

def fig4_effects(models):
    """
    Per model and attack family, joint pass rate — fraction of scenario pairs
    where the agent acted on the legitimate variant AND refused the
    unauthorized variant. Four bars per model: baseline + 3 primitives.
    Asterisks above each non-baseline bar mark McNemar significance vs
    baseline (paired-binomial exact, two-sided), tested per pair on the
    joint-pass indicator.
    """
    fig, axes = plt.subplots(1, len(TYPES), figsize=(4.6 * len(TYPES), 5.2),
                             sharey=True, squeeze=False,
                             gridspec_kw={"wspace": 0.06})

    n_m = len(models)
    cond_x = np.arange(len(CONDITIONS))
    bar_w = 0.82 / n_m
    offsets = (np.arange(n_m) - (n_m - 1) / 2) * bar_w

    closed_idxs = [i for i, m in enumerate(models)
                   if SOURCE_TYPE.get(m["slug"]) == "closed"]
    open_idxs = [i for i, m in enumerate(models)
                 if SOURCE_TYPE.get(m["slug"]) == "open"]

    for ci, atype in enumerate(TYPES):
        ax = axes[0, ci]
        var_t, var_u = VARIANT_PAIR[atype]

        for mi, m in enumerate(models):
            xs = cond_x + offsets[mi]
            _, base_joint = joint_pass_vector(m, atype, var_t, var_u, "baseline")
            base_rate = base_joint.mean() * 100 if len(base_joint) else 0.0
            heights, hatches = [base_rate], [None]
            for cond in CONDITIONS[1:]:
                eff = joint_paired_effect(m, atype, var_t, var_u, "baseline", cond)
                if eff is None:
                    heights.append(0.0); hatches.append(None); continue
                heights.append(eff["rate_b"])
                hatches.append(sig_hatch(eff["p"]))

            color = MODEL_COLORS[mi % len(MODEL_COLORS)]
            for x_i, h, hatch in zip(xs, heights, hatches):
                ax.bar(x_i, h, bar_w,
                       label=m["display"] if x_i == xs[0] else None,
                       color=color, hatch=hatch,
                       edgecolor="black", linewidth=0.4)

        ax.set_xticks(cond_x)
        ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS], fontsize=17)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_ylim(0, 80)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        head, _ = TYPE_TITLE[atype]
        ax.set_title(head, fontsize=19, fontweight="bold")
        if ci == 0:
            ax.set_ylabel("Both variants correct (%)", fontsize=16)

    # Single bounded legend with "CLOSED SOURCE" / "OPEN SOURCE" row labels on
    # the left. Column-major ncol=6: column 0 holds the row-label headers,
    # columns 1-5 hold paired model entries (closed top, open bottom).
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    closed_handles = [handle_by_label[models[i]["display"]] for i in closed_idxs]
    open_handles   = [handle_by_label[models[i]["display"]] for i in open_idxs]
    closed_labels  = [models[i]["display"] for i in closed_idxs]
    open_labels    = [models[i]["display"] for i in open_idxs]

    blank = Patch(facecolor="none", edgecolor="none", linewidth=0)
    ordered_h = [blank, blank]
    ordered_l = ["CLOSED SOURCE", "OPEN SOURCE"]
    for c_h, o_h, c_l, o_l in zip(closed_handles, open_handles,
                                  closed_labels, open_labels):
        ordered_h.extend([c_h, o_h])
        ordered_l.extend([c_l, o_l])

    fig.legend(ordered_h, ordered_l,
               loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.0),
               frameon=True, edgecolor="#888", fancybox=False,
               fontsize=15,
               handlelength=2.0, handletextpad=0.6,
               columnspacing=1.4, borderpad=0.7, labelspacing=0.6)
    fig.subplots_adjust(bottom=0.26, top=0.92, left=0.06, right=0.99)

    out = FIG_DIR / "fig4_effects.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Fig 4 — transitive 2-hop: where first-hop primitives stop working
# ---------------------------------------------------------------------------

CONDS_TRANSITIVE = ["both"]
COND_LABEL_TRANSITIVE = {
    "both": "both",
}


def _variant_pair_n10(model, type_, var_a, var_b, condition):
    """
    Count McNemar paired outcomes where (var_a, var_b) = (tool_called, tool_blocked)
    — i.e., the model differentiated between the two variants in the correct
    direction. Returns (n10, n01, n_pairs) or (0, 0, 0) if cell missing.
    """
    by_pair: dict[int, dict[str, str]] = {}
    cells = model.get("raw_records") or []
    for r in cells:
        if r["type"] != type_ or r["condition"] != condition:    continue
        if r["variant"] not in (var_a, var_b):                   continue
        sid = r["scenario"]
        pid = _pair_id(sid)
        if pid is None: continue
        by_pair.setdefault(pid, {})[r["variant"]] = r["actual"]
    n10 = n01 = n_pairs = 0
    for d in by_pair.values():
        if var_a not in d or var_b not in d: continue
        n_pairs += 1
        a, b = d[var_a], d[var_b]
        if a == "tool_called"  and b == "tool_blocked": n10 += 1
        elif a == "tool_blocked" and b == "tool_called": n01 += 1
    return n10, n01, n_pairs


def fig6_transitive(models):
    """
    Schematic comparing the paired structure of source-label IPI and
    scope-check IPI. Per-model joint-pass numbers for these families are
    already shown in fig4_effects; this figure shows only what flips
    between the two halves of a pair on each family.
    """
    schematic_path = FIG_DIR / "sudobench_ipi_families.png"
    fig = plt.figure(figsize=(12.0, 2.9))
    ax_img = fig.add_subplot(1, 1, 1)
    if schematic_path.exists():
        ax_img.imshow(mpimg.imread(str(schematic_path)))
    ax_img.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    out = FIG_DIR / "fig6_transitive.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out



# ---------------------------------------------------------------------------
# Fig 3 — safety–utility Pareto (MAIN)
# ---------------------------------------------------------------------------

def fig5_utility(models):
    """
    Safety–utility Pareto. X = refusal rate on untrusted attacks (safety),
    Y = pass rate on legitimate requests (utility). Per model, an arrow
    goes from the baseline (no primitives) to the both condition. The
    Pareto frontier over the `both` endpoints is drawn for reference.
    Compliant-but-unsafe models sit in the bottom-right; the frontier
    highlights which models genuinely buy defence without sacrificing
    utility.
    """
    # Three-panel detailed version (one per family), appendix figure.
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    panels = [("source_label_ipi", "Source-label IPI"),
              ("scope_check_ipi",  "Scope-check IPI"),
              ("confused_deputy",  "Confused deputy")]

    def _safety(m, cond, atype):
        _, refuse_var = VARIANT_PAIR[atype]
        c = m["cells"].get((cond, atype, refuse_var))
        return (c["pass_rate"] * 100) if c and c.get("n") else 0.0

    def _utility(m, cond, atype):
        act_var, _ = VARIANT_PAIR[atype]
        c = m["cells"].get((cond, atype, act_var))
        return (c["pass_rate"] * 100) if c and c.get("n") else 0.0

    for ai, (atype, title) in enumerate(panels):
        ax = axes[ai]
        endpoints = []        # (safety, utility, color, name) at per-model frontier tip
        all_points = []       # (safety, utility) across every (model, cond) cell
        for mi, m in enumerate(models):
            color = MODEL_COLORS[mi % len(MODEL_COLORS)]
            cond_pts = {cond: (_safety(m, cond, atype), _utility(m, cond, atype))
                        for cond in CONDITIONS}
            all_points.extend(cond_pts.values())

            # Per-model Pareto frontier across 4 conditions (non-dominated in
            # safety AND utility). Order: ascending safety, then ascending
            # utility. The arrow trajectory follows this frontier from
            # baseline-side outward.
            pm_pts = sorted(cond_pts.items(),
                            key=lambda kv: (kv[1][0], kv[1][1]))
            pm_frontier = []
            best_u = -1.0
            for cond, (s, u) in pm_pts:
                if u > best_u:
                    pm_frontier.append((cond, s, u))
                    best_u = u
            # Always anchor the trajectory at baseline so the arrow shows the
            # model's improvement from "no primitives" along the frontier.
            base_s, base_u = cond_pts["baseline"]
            traj = [("baseline", base_s, base_u)] + [
                p for p in pm_frontier if p[0] != "baseline"
            ]

            # Polyline through the per-model frontier
            xs = [p[1] for p in traj]; ys = [p[2] for p in traj]
            ax.plot(xs, ys, color=color, lw=1.6, alpha=0.85, zorder=2,
                    solid_capstyle="round")
            # Arrow head on the final segment
            if len(traj) >= 2:
                ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                            arrowprops=dict(arrowstyle="->", color=color,
                                            lw=1.6, alpha=0.85),
                            zorder=2)

            # Markers: open at baseline, small filled at intermediate frontier
            # stops, large filled at the final (best) frontier endpoint.
            ax.plot([base_s], [base_u], marker="o", markersize=7, color=color,
                    markerfacecolor="white", markeredgewidth=1.6, zorder=3)
            for cond, s, u in traj[1:-1]:
                ax.plot([s], [u], marker="o", markersize=6, color=color,
                        markeredgecolor=color, markeredgewidth=0.6, zorder=3)
            if len(traj) >= 2:
                tip_s, tip_u = traj[-1][1], traj[-1][2]
                ax.plot([tip_s], [tip_u], marker="o", markersize=9, color=color,
                        markeredgecolor="black", markeredgewidth=0.6, zorder=3)
                endpoints.append((tip_s, tip_u, color, m["display"]))
            else:
                endpoints.append((base_s, base_u, color, m["display"]))

        # Pareto frontier over ALL (model, condition) points (40 per panel).
        pts = sorted(set(all_points), key=lambda p: (-p[0], -p[1]))
        frontier, best_u = [], -1
        for s, u in pts:
            if u > best_u:
                frontier.append((s, u)); best_u = u
        if len(frontier) > 1:
            fs = [p[0] for p in frontier]
            fu = [p[1] for p in frontier]
            ax.plot(fs, fu, ls="--", color="#555555", lw=1.2, alpha=0.6,
                    zorder=1, label="Pareto frontier (all conditions)")

        # Per-model labels are rendered through a shared legend below the
        # figure rather than in-panel; the upper-right cluster on
        # scope-check and confused-deputy makes per-arrow text labels
        # collide irretrievably.

        ax.set_xlim(8, 112)
        ax.set_ylim(22, 100)
        ax.set_xticks(range(20, 101, 20))
        ax.set_yticks(range(30, 101, 10))
        ax.tick_params(axis="both", labelsize=11)
        ax.set_title(title, fontsize=13)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_xlabel("Attacks refused (%)", fontsize=12)
        if ai == 0:
            ax.set_ylabel("Legitimate requests handled (%)", fontsize=12)

    # Shared legend across panels, structured like fig4's: closed on row 1,
    # open on row 2, frontier-line indicator in a separate legend block.
    model_handles = [
        plt.Line2D([], [], marker="o", linestyle="-",
                   color=MODEL_COLORS[mi % len(MODEL_COLORS)],
                   markerfacecolor=MODEL_COLORS[mi % len(MODEL_COLORS)],
                   markeredgecolor="black", markeredgewidth=0.4,
                   markersize=8, lw=1.6, label=m["display"])
        for mi, m in enumerate(models)
    ]
    closed_idxs = [i for i, m in enumerate(models)
                   if SOURCE_TYPE.get(m["slug"]) == "closed"]
    open_idxs   = [i for i, m in enumerate(models)
                   if SOURCE_TYPE.get(m["slug"]) == "open"]
    closed_h = [model_handles[i] for i in closed_idxs]
    open_h   = [model_handles[i] for i in open_idxs]
    closed_l = [models[i]["display"] for i in closed_idxs]
    open_l   = [models[i]["display"] for i in open_idxs]

    blank = Patch(facecolor="none", edgecolor="none", linewidth=0)
    ordered_h = [blank, blank]
    ordered_l = ["CLOSED SOURCE", "OPEN SOURCE"]
    for c_h, o_h, c_l, o_l in zip(closed_h, open_h, closed_l, open_l):
        ordered_h.extend([c_h, o_h])
        ordered_l.extend([c_l, o_l])

    fig.legend(ordered_h, ordered_l,
               loc="lower center", ncol=6,
               bbox_to_anchor=(0.42, -0.03),
               frameon=True, edgecolor="#888", fancybox=False,
               fontsize=11,
               handlelength=2.0, handletextpad=0.5,
               columnspacing=1.4, borderpad=0.7, labelspacing=0.6)

    fig.legend(handles=[plt.Line2D([], [], ls="--", color="#555555",
                                    lw=1.3)],
               labels=["Pareto frontier (all conditions)"],
               loc="lower right", bbox_to_anchor=(0.99, -0.03),
               frameon=True, edgecolor="#888", fancybox=False,
               fontsize=11, handlelength=2.0, handletextpad=0.5,
               borderpad=0.7)
    fig.subplots_adjust(bottom=0.22, top=0.92, left=0.05, right=0.99)

    out = FIG_DIR / "fig_utility_full.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def fig5_pareto(models):
    """
    Single-panel collapsed safety-utility Pareto for the main paper.

    All 30 (model, family) best-condition operating points on a single
    safety-utility plane. Color and shape both encode family; condition
    is not surfaced (the dot is just "the model's best operating point
    on that family"). One Pareto frontier per family connects each
    family's non-dominated points. Gray ×'s are baseline operating
    points across all 30 cells.
    """
    FAMILY_COLOR = {
        "source_label_ipi":  "#2c7a7b",   # teal
        "scope_check_ipi":   "#5d3a87",   # purple
        "confused_deputy":   "#3182ce",   # blue
    }
    FAMILY_MARKER = {
        "source_label_ipi":  "^",   # triangle up
        "scope_check_ipi":   "o",   # circle
        "confused_deputy":   "s",   # square
    }
    FAMILY_LABEL = {
        "source_label_ipi":  "Source-label IPI",
        "scope_check_ipi":   "Scope-check IPI",
        "confused_deputy":   "Confused deputy",
    }

    def _safety(m, cond, atype):
        _, rv = VARIANT_PAIR[atype]
        c = m["cells"].get((cond, atype, rv))
        return (c["pass_rate"] * 100) if c and c.get("n") else 0.0

    def _utility(m, cond, atype):
        av, _ = VARIANT_PAIR[atype]
        c = m["cells"].get((cond, atype, av))
        return (c["pass_rate"] * 100) if c and c.get("n") else 0.0

    def _joint(m, cond, atype):
        av, rv = VARIANT_PAIR[atype]
        _, j = joint_pass_vector(m, atype, av, rv, cond)
        return j.mean() if len(j) else 0.0

    fig, ax = plt.subplots(figsize=(6.2, 4.8))

    by_family: dict[str, list[tuple[float, float]]] = {a: [] for a in FAMILY_COLOR}
    baseline_pts: list[tuple[float, float]] = []

    for m in models:
        for atype in FAMILY_COLOR:
            baseline_pts.append((_safety(m, "baseline", atype),
                                 _utility(m, "baseline", atype)))
            best_cond, best_j = None, -1.0
            for cond in ("tags", "capabilities", "both"):
                j = _joint(m, cond, atype)
                if j > best_j:
                    best_j, best_cond = j, cond
            by_family[atype].append(
                (_safety(m, best_cond, atype),
                 _utility(m, best_cond, atype))
            )

    # Baseline cloud
    if baseline_pts:
        bxs = [p[0] for p in baseline_pts]
        bys = [p[1] for p in baseline_pts]
        ax.scatter(bxs, bys, s=14, c="#bfbfbf", marker="x",
                   linewidths=0.8, alpha=0.65, zorder=2)

    # Per-family points + per-family Pareto frontier
    for atype, pts in by_family.items():
        color = FAMILY_COLOR[atype]
        marker = FAMILY_MARKER[atype]
        if pts:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.scatter(xs, ys, s=48, c=color, marker=marker,
                       edgecolors="black", linewidths=0.4, alpha=0.9,
                       zorder=3)
            # Family-specific Pareto frontier across its 10 best-points
            sorted_pts = sorted(set(pts), key=lambda p: (-p[0], -p[1]))
            frontier, best_u = [], -1
            for s, u in sorted_pts:
                if u > best_u:
                    frontier.append((s, u)); best_u = u
            if len(frontier) > 1:
                fs = [p[0] for p in frontier]; fu = [p[1] for p in frontier]
                ax.plot(fs, fu, ls="--", color=color, lw=1.3, alpha=0.7,
                        zorder=1)

    # Ideal-direction annotation
    ax.annotate("ideal", xy=(99, 99), xytext=(86, 86),
                ha="right", va="top",
                fontsize=9, color="#333333", fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color="#333333",
                                lw=1.0, alpha=0.85), zorder=4)

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Attacks refused (%)", fontsize=10)
    ax.set_ylabel("Legitimate requests handled (%)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    family_handles = [
        plt.Line2D([], [], marker=FAMILY_MARKER[a], linestyle="",
                   markerfacecolor=FAMILY_COLOR[a], markeredgecolor="black",
                   markeredgewidth=0.4, markersize=8, label=FAMILY_LABEL[a])
        for a in FAMILY_COLOR
    ]
    baseline_h = plt.Line2D([], [], marker="x", linestyle="",
                            color="#bfbfbf", markersize=7,
                            label="baseline (no primitives)")
    frontier_h = plt.Line2D([], [], ls="--", color="#555555",
                            lw=1.3, label="per-family Pareto frontier")

    ax.legend(handles=family_handles + [frontier_h, baseline_h],
              loc="lower right", fontsize=8.5,
              framealpha=0.9, frameon=True, edgecolor="#bbb",
              handlelength=1.4, borderpad=0.5)

    out = FIG_DIR / "fig5_utility.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    models = discover_models()
    if not models:
        print("No models found in results/")
        return
    print(f"Found {len(models)} models: {[m['display'] for m in models]}")
    # fig5_pareto (single-panel collapsed Pareto) is cut from the main paper;
    # the safety-utility story is carried by the per-family detail figure
    # below (fig5_utility, output as fig_utility_full.pdf, appendix).
    for fn in (fig1_headroom, fig4_effects, fig5_utility, fig6_transitive):
        out = fn(models)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
