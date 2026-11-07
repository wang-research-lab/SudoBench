#!/usr/bin/env python3
"""
appendix_tables.py — Regenerate the per-cell appendix LaTeX tables.

Reads every results/<model>/eval_records.jsonl, computes per-pair
joint-pass indicators, runs paired McNemar exact tests vs the no-primitives
baseline, and writes:

  Formatting_Instructions_For_NeurIPS_2026/tables/tab_effects.tex
    - Δ joint-pass rate (pp) vs baseline, per model × condition × attack family.

  Formatting_Instructions_For_NeurIPS_2026/tables/tab_appendix_mde.tex
    - Per-cell n10/n01, Δ with bootstrap 95% CI, McNemar exact MDE@80%,
      noise fraction min(n10,n01)/(n10+n01), and exact two-sided p.

  Formatting_Instructions_For_NeurIPS_2026/tables/tab_prehoc_power.tex
    - Pre-hoc sensitivity table: at N=45 pairs and α=.05, for each plausible
      discordance count n_d, the minimum |n10−n01| needed to reject H0 and
      the corresponding MDE in proportion-difference and percentage-point
      terms at 80% power.

Each (model, attack-family, condition) cell pairs every scenario pair across
baseline and the condition under test. A pair joint-passes when the agent
acts on the trusted/in-scope variant AND refuses the untrusted/out-of-scope
variant. The McNemar exact two-sided p is computed on the joint-pass
indicator paired across baseline and the test condition.

Usage
-----
  python appendix_tables.py
"""
import json, os
from pathlib import Path

import numpy as np
from scipy import stats


BENCH_DIR      = Path(__file__).parent
RESULTS_DIR    = BENCH_DIR / "results"
TABLES_DIR     = Path(os.environ.get("SUDOBENCH_TABLES_DIR", BENCH_DIR / "tables"))

# ---------------------------------------------------------------------------
# Display configuration
# ---------------------------------------------------------------------------
MODEL_ORDER = [
    "claude-sonnet-4-6", "gemini-gemini-3.1-pro-preview",
    "openrouter-openai-gpt-5.4", "openrouter-qwen-qwen3.6-plus",
    "openrouter-x-ai-grok-4.20",
    "openrouter-deepseek-deepseek-v3.2", "openrouter-z-ai-glm-5.1",
    "openrouter-moonshotai-kimi-k2.6", "openrouter-xiaomi-mimo-v2.5-pro",
    "openrouter-minimax-minimax-m2.7",
]
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

# Attack family → (key, label, attack_type, execute_variant, refuse_variant)
FAMILIES = [
    ("source_label_ipi",     "Source-label IPI",     "source_label_ipi",
                                                     "trusted",  "untrusted"),
    ("scope_check_ipi", "Scope-check IPI", "scope_check_ipi",
                                                     "in_scope", "out_of_scope"),
    ("cd",                   "Confused deputy",      "confused_deputy",
                                                     "trusted",  "untrusted"),
]

# Conditions evaluated against baseline for each family.
FAMILY_CONDITIONS = {
    "source_label_ipi":     ["tags", "capabilities", "both"],
    "cd":                   ["tags", "capabilities", "both"],
    "scope_check_ipi":      ["tags", "capabilities", "both"],
}

COND_LABEL = {  # display names used in the paper
    "tags":         "\\textsc{source}",
    "capabilities": "\\textsc{scopes}",
    "both":         "\\textsc{full}",
}

# All conditions that ever appear in tab_effects, in display order.
ALL_CONDITIONS = ["tags", "capabilities", "both"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(slug: str) -> dict[tuple, dict]:
    """Return {(scenario, condition): {"passed": bool, "type": str, "variant": str}}."""
    rp = RESULTS_DIR / slug / "eval_records.jsonl"
    if not rp.exists():
        return {}
    out: dict[tuple, dict] = {}
    for line in rp.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[(r["scenario"], r.get("condition", "baseline"))] = {
            "passed":  bool(r["passed"]),
            "type":    r["type"],
            "variant": r["variant"],
        }
    return out


def pair_id_of(scenario: str) -> str:
    for suf in ("_transitive_2hop_trusted", "_transitive_2hop",
                "_out_of_scope", "_in_scope",
                "_trusted", "_untrusted"):
        if scenario.endswith(suf):
            return scenario[: -len(suf)]
    return scenario


def joint_pass_vector(records: dict[tuple, dict],
                      attack_type: str,
                      trusted_variant: str,
                      untrusted_variant: str,
                      condition: str) -> dict[str, bool]:
    """Return {pair_id: joint_passed} for one (family, condition) cell."""
    out: dict[str, bool] = {}
    for (scen, cond), info in records.items():
        if cond != condition:
            continue
        if info["type"] != attack_type:
            continue
        if info["variant"] != trusted_variant:
            continue
        other_scen = scen[: -len("_" + trusted_variant)] + "_" + untrusted_variant
        other = records.get((other_scen, cond))
        if other is None or other["type"] != attack_type or other["variant"] != untrusted_variant:
            continue
        pid = pair_id_of(scen)
        out[pid] = info["passed"] and other["passed"]
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mcnemar_exact_p(n10: int, n01: int) -> float | None:
    """Two-sided exact McNemar (binomial) p-value on discordant pairs."""
    n = n10 + n01
    if n == 0:
        return None
    # Two-sided exact binomial test, p=0.5
    return float(stats.binomtest(min(n10, n01), n, 0.5, alternative="two-sided").pvalue)


def mcnemar_reject_set(n_d: int, alpha: float = 0.05) -> set[int]:
    """Exact two-sided rejection set for H0: P(n10 | discordant) = 0.5."""
    if n_d == 0:
        return set()
    return {k for k in range(n_d + 1)
            if stats.binomtest(k, n_d, 0.5, alternative="two-sided").pvalue <= alpha}


def mcnemar_min_reject_k(n_d: int, alpha: float = 0.05) -> int | None:
    """Smallest |n10 - n01| at which H0 is rejected at level alpha.

    For each k in the rejection set, |n10 - n01| = |2k - n_d|; we want the
    minimum over k in rs.
    """
    rs = mcnemar_reject_set(n_d, alpha)
    if not rs:
        return None
    return min(abs(2 * k - n_d) for k in rs)


def mcnemar_exact_mde_skew(n_d: int,
                           target_power: float = 0.80,
                           alpha: float = 0.05,
                           grid: int = 1001) -> float | None:
    """Smallest skew δ = 2|true_p − 0.5| (proportion-diff among discordants)
    such that the exact McNemar test has ≥ target_power at level alpha.

    Returns δ ∈ [0, 1]; δ = 0.4 means true 70/30 split among discordants.
    """
    if n_d == 0:
        return None
    rs = mcnemar_reject_set(n_d, alpha)
    if not rs:
        return None
    for step in range(grid):
        true_p = 0.5 + step * (0.5 / (grid - 1))
        power = float(sum(stats.binom.pmf(k, n_d, true_p) for k in rs))
        if power >= target_power:
            return round(2 * (true_p - 0.5), 4)
    return None


def bootstrap_delta_ci(base_vec: list[bool],
                       cond_vec: list[bool],
                       n_boot: int = 5000,
                       seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI on (cond_rate - base_rate)."""
    rng = np.random.default_rng(seed)
    base = np.asarray(base_vec, dtype=int)
    cond = np.asarray(cond_vec, dtype=int)
    n = len(base)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = cond[idx].mean(1) - base[idx].mean(1)
    return (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def mde_pp_exact(n_pairs: int, n_disc: int,
                 target_power: float = 0.80, alpha: float = 0.05) -> float | None:
    """MDE in proportion-difference (rate-diff) terms, in [0, 1], at observed n_disc.

    Uses the exact-binomial McNemar test (more accurate than normal-approx
    at small n). Translates the discordant-pair skew δ to overall rate-diff
    via δ_rate = δ_skew · (n_disc / N), assuming n_disc is fixed.
    """
    if n_pairs == 0 or n_disc == 0:
        return None
    skew = mcnemar_exact_mde_skew(n_disc, target_power=target_power, alpha=alpha)
    if skew is None:
        return None
    return skew * (n_disc / n_pairs)


def noise_fraction(n10: int, n01: int) -> float | None:
    """Fraction of discordant pairs flipping against the dominant direction.

    0   = perfectly monotone (all flips help one side),
    0.5 = pure noise (flips cancel),
    None = no discordance.
    """
    n_d = n10 + n01
    if n_d == 0:
        return None
    return min(n10, n01) / n_d


def odds_ratio_corrected(n10: int, n01: int) -> float | None:
    """McNemar conditional odds ratio with Haldane-Anscombe correction.

    OR = (n01 + 0.5) / (n10 + 0.5). The correction handles cells with zero
    counts (which occur whenever a condition strictly improves or strictly
    degrades joint pass) without producing infinities. Returns None when
    both discordance counts are zero.
    """
    if n10 == 0 and n01 == 0:
        return None
    return (n01 + 0.5) / (n10 + 0.5)


def sig_stars(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "$^{***}$"
    if p < 0.01:
        return "$^{**}$"
    if p < 0.05:
        return "$^{*}$"
    return ""


# ---------------------------------------------------------------------------
# Cell statistics
# ---------------------------------------------------------------------------

def variant_cell_stats(records: dict[tuple, dict],
                       attack_type: str,
                       variant: str,
                       condition: str) -> dict | None:
    """Paired McNemar on the per-scenario pass indicator for a single variant.

    Compares the agent's pass rate on `variant` (e.g., "trusted" or "untrusted")
    under `condition` vs `baseline`. Δ > 0 means the condition increases the
    rate of correct outcomes on this variant.
    """
    pairs: dict[str, dict[str, bool]] = {}
    for (scen, cond), info in records.items():
        if info["type"] != attack_type or info["variant"] != variant:
            continue
        pairs.setdefault(scen, {})[cond] = info["passed"]

    n11 = n10 = n01 = n00 = 0
    for scen, cs in pairs.items():
        if "baseline" not in cs or condition not in cs:
            continue
        b, c = cs["baseline"], cs[condition]
        if b and c:           n11 += 1
        elif b and not c:     n10 += 1
        elif (not b) and c:   n01 += 1
        else:                 n00 += 1
    n = n00 + n01 + n10 + n11
    if n == 0:
        return None

    base_rate = (n10 + n11) / n
    cond_rate = (n01 + n11) / n
    delta = cond_rate - base_rate
    p = mcnemar_exact_p(n10, n01)
    mde = mde_pp_exact(n, n10 + n01)
    return {
        "n":         n,
        "n10":       n10,
        "n01":       n01,
        "base_rate": base_rate,
        "cond_rate": cond_rate,
        "delta":     delta,
        "p":         p,
        "mde":       mde,
    }


def cell_stats(records: dict[tuple, dict],
               family_key: str,
               condition: str) -> dict | None:
    """Compute joint-pass paired statistics for one (family, condition) cell."""
    _, _, atype, tv, uv = next(f for f in FAMILIES if f[0] == family_key)
    base = joint_pass_vector(records, atype, tv, uv, "baseline")
    cond = joint_pass_vector(records, atype, tv, uv, condition)
    pairs = sorted(set(base) & set(cond))
    if not pairs:
        return None

    base_v = [base[p] for p in pairs]
    cond_v = [cond[p] for p in pairs]
    n11 = sum(1 for b, c in zip(base_v, cond_v) if b and c)
    n10 = sum(1 for b, c in zip(base_v, cond_v) if b and not c)
    n01 = sum(1 for b, c in zip(base_v, cond_v) if (not b) and c)
    n00 = sum(1 for b, c in zip(base_v, cond_v) if (not b) and (not c))
    n = len(pairs)

    base_rate = sum(base_v) / n
    cond_rate = sum(cond_v) / n
    delta     = cond_rate - base_rate
    ci_lo, ci_hi = bootstrap_delta_ci(base_v, cond_v)
    p = mcnemar_exact_p(n10, n01)
    mde = mde_pp_exact(n, n10 + n01)
    noise = noise_fraction(n10, n01)
    odds = odds_ratio_corrected(n10, n01)

    return {
        "n":         n,
        "n00":       n00,
        "n01":       n01,
        "n10":       n10,
        "n11":       n11,
        "base_rate": base_rate,
        "cond_rate": cond_rate,
        "delta":     delta,
        "ci_lo":     ci_lo,
        "ci_hi":     ci_hi,
        "p":         p,
        "mde":       mde,
        "noise":     noise,
        "odds":      odds,
    }


# ---------------------------------------------------------------------------
# Table 1: tab_effects.tex
# ---------------------------------------------------------------------------

def fmt_delta(stats: dict | None) -> str:
    if stats is None:
        return "---"
    pp = stats["delta"] * 100
    return f"{pp:+.1f}{sig_stars(stats['p'])}"


def build_effects_table(per_model: dict[str, dict[tuple, dict]]) -> str:
    out = []
    out.append("\\begin{table}[H]")
    out.append("    \\centering")
    out.append("    \\footnotesize")
    out.append("    \\caption{Joint-pass-rate $\\Delta$ (percentage points) vs.\\ "
               "no-primitives baseline, per model $\\times$ condition $\\times$ attack "
               "family. Pairs are matched by scenario across conditions; significance is "
               "exact two-sided McNemar on the per-pair joint-pass indicator. "
               "$^{*}\\,p<.05$, $^{**}\\,p<.01$, $^{***}\\,p<.001$. Em-dash cells are "
               "conditions not evaluated for that family.}")
    out.append("    \\label{tab:effects}")
    out.append("    \\begin{tabular}{llrrr}")
    out.append("        \\toprule")
    out.append("        Model & Condition & Source-label IPI & Scope-check IPI & Confused deputy \\\\")
    out.append("        \\midrule")

    for slug in MODEL_ORDER:
        if slug not in per_model:
            continue
        records = per_model[slug]
        label = MODEL_LABEL.get(slug, slug)
        first = True
        for cond in ALL_CONDITIONS:
            row_cells = []
            for fam_key in ("source_label_ipi", "scope_check_ipi", "cd"):
                if cond not in FAMILY_CONDITIONS[fam_key]:
                    row_cells.append("---")
                    continue
                stats_cell = cell_stats(records, fam_key, cond)
                row_cells.append(fmt_delta(stats_cell))
            if all(c == "---" for c in row_cells):
                continue
            model_cell = label if first else ""
            first = False
            out.append(f"        {model_cell:<14} & {COND_LABEL[cond]:<18} "
                       f"& {row_cells[0]:>8} & {row_cells[1]:>8} & {row_cells[2]:>8} \\\\")
        out.append("        \\midrule")

    # drop trailing midrule
    if out[-1] == "        \\midrule":
        out.pop()
    out.append("        \\bottomrule")
    out.append("    \\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Table 2: tab_appendix_mde.tex
# ---------------------------------------------------------------------------

def fmt_pct(x: float) -> str:
    return f"{x*100:+.1f}"


def fmt_pct_unsigned(x: float) -> str:
    return f"{x*100:.1f}"


def fmt_p(p: float | None) -> str:
    if p is None:
        return "---"
    if p < 0.001:
        return "\\mbox{$<$.001}"
    return f"{p:.3f}"


def _row_color(stats_cell: dict) -> str:
    """Background color command for a row.

    Green   — observed |delta| meets or exceeds the cell's MDE at 80% power.
              The effect is both significant and adequately powered.
    Yellow  — McNemar-significant at alpha=0.05 but observed |delta| is
              below MDE; the test rejected H0 but the effect lies below
              the cell's pre-specified detection floor (interpret with care).
    Unshaded — not significant, or no rejection region exists for this
              discordance count.
    """
    p   = stats_cell.get("p")
    mde = stats_cell.get("mde")
    delta = stats_cell.get("delta")
    if mde is not None and delta is not None and abs(delta) >= mde:
        return "\\rowcolor{green!22}$\\bullet$\\,"
    if p is not None and p < 0.05:
        return "\\rowcolor{yellow!28}$\\circ$\\,"
    return ""


def build_mde_table(per_model: dict[str, dict[tuple, dict]]) -> str:
    out = []
    out.append("\\begin{small}")
    out.append("\\begin{xltabular}{\\linewidth}{@{\\extracolsep{\\fill}}lllrrrrrrrr@{}}")
    out.append("    \\caption{Per-cell joint-pass-rate effect with paired statistical "
               "diagnostics. All cells use $N=45$ paired scenarios. Row shading reflects "
               "the joint power analysis: \\colorbox{green!22}{green} marks cells where "
               "$|\\Delta|\\ge \\text{MDE}$ at $80\\%$ power (the effect is both "
               "McNemar-significant and above the pre-specified detection floor); "
               "\\colorbox{yellow!28}{yellow} marks cells that are McNemar-significant at "
               "$\\alpha=0.05$ but with $|\\Delta|<\\text{MDE}$ (rejection of $H_0$ at an "
               "effect size below the design-stage detection floor; interpret with care); "
               "unshaded rows are not significant or have no rejection region for the observed "
               "discordance. The left block reports the effect: $\\Delta$ in percentage points "
               "and the Haldane-Anscombe-corrected McNemar odds ratio "
               "$\\mathrm{OR}=(n_{01}+0.5)/(n_{10}+0.5)$, with a 95\\% percentile bootstrap "
               "confidence interval on $\\Delta$ (5{,}000 resamples). The right block reports "
               "the test diagnostics: McNemar discordance counts $n_{10}$ (pass at baseline "
               "only) and $n_{01}$ (pass at condition only); the minimum detectable effect at "
               "$80\\%$ power (MDE) under the exact two-sided binomial McNemar test, expressed "
               "on the rate-difference scale at the observed discordance (reported as $\\infty$ "
               "when the exact test has no rejection region at $\\alpha=0.05$, which occurs "
               "for $n_{10}+n_{01}\\le 5$); the noise fraction "
               "$f_{\\!N}=\\min(n_{10},n_{01})/(n_{10}+n_{01})$, where $0$ denotes unanimous "
               "direction of discordant flips and $0.5$ denotes balanced disagreement; and "
               "the exact two-sided McNemar $p$-value.}\\label{tab:appendix-mde} \\\\")
    out.append("    \\toprule")
    out.append("                   &        &           "
               "& \\multicolumn{3}{c}{Effect} "
               "& \\multicolumn{5}{c}{Statistical testing} \\\\")
    out.append("    \\cmidrule(lr){4-6} \\cmidrule(lr){7-11}")
    out.append("    Model & Family & Condition "
               "& $\\Delta$ (pp) & 95\\% CI & OR "
               "& $n_{10}$ & $n_{01}$ & MDE@80\\% & $f_{\\!N}$ & $p$ \\\\")
    out.append("    \\midrule")
    out.append("    \\endfirsthead")
    out.append("    \\multicolumn{11}{l}{\\textit{Table~\\ref{tab:appendix-mde} (continued)}} \\\\")
    out.append("    \\toprule")
    out.append("                   &        &           "
               "& \\multicolumn{3}{c}{Effect} "
               "& \\multicolumn{5}{c}{Statistical testing} \\\\")
    out.append("    \\cmidrule(lr){4-6} \\cmidrule(lr){7-11}")
    out.append("    Model & Family & Condition "
               "& $\\Delta$ (pp) & 95\\% CI & OR "
               "& $n_{10}$ & $n_{01}$ & MDE@80\\% & $f_{\\!N}$ & $p$ \\\\")
    out.append("    \\midrule")
    out.append("    \\endhead")
    out.append("    \\midrule \\multicolumn{11}{r}{\\textit{Continued on next page}} \\\\")
    out.append("    \\endfoot")
    out.append("    \\bottomrule")
    out.append("    \\endlastfoot")

    for slug in MODEL_ORDER:
        if slug not in per_model:
            continue
        records = per_model[slug]
        label = MODEL_LABEL.get(slug, slug)
        first_model_row = True
        for fam_key, fam_label, _atype, _tv, _uv in FAMILIES:
            for cond in FAMILY_CONDITIONS[fam_key]:
                stats_cell = cell_stats(records, fam_key, cond)
                if stats_cell is None:
                    continue
                model_cell = label if first_model_row else ""
                first_model_row = False
                ci = f"[{fmt_pct(stats_cell['ci_lo'])}, {fmt_pct(stats_cell['ci_hi'])}]"
                mde_str = (fmt_pct_unsigned(stats_cell["mde"])
                           if stats_cell["mde"] is not None else "$\\infty$")
                noise_str = f"{stats_cell['noise']:.2f}" if stats_cell["noise"] is not None else "---"
                or_str = f"{stats_cell['odds']:.2f}" if stats_cell["odds"] is not None else "---"
                color = _row_color(stats_cell)
                out.append(
                    f"    {color}{model_cell:<14} & {fam_label:<10} & {COND_LABEL[cond]:<18} "
                    f"& {fmt_pct(stats_cell['delta']):>6} & {ci:>16} & {or_str:>5} "
                    f"& {stats_cell['n10']:>3} & {stats_cell['n01']:>3} "
                    f"& {mde_str:>8} & {noise_str:>5} & {fmt_p(stats_cell['p']):>8} \\\\"
                )
        out.append("    \\midrule")
    if out[-1] == "    \\midrule":
        out.pop()
    out.append("\\end{xltabular}")
    out.append("\\end{small}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Table 3: tab_prehoc_power.tex
# ---------------------------------------------------------------------------

PREHOC_N_PAIRS = 45  # paired scenarios per (model, family, condition) cell
PREHOC_N_DISC_GRID = [3, 5, 6, 7, 10, 15, 20, 25, 30, 35, 40, 45]


def build_prehoc_power_table() -> str:
    out = []
    out.append("\\begin{table}[H]")
    out.append("    \\centering")
    out.append("    \\footnotesize")
    out.append("    \\caption{Pre-hoc sensitivity of the per-cell test. "
               f"At $N={PREHOC_N_PAIRS}$ paired scenarios, $\\alpha=0.05$, target power "
               "$0.80$, and an exact two-sided binomial McNemar test, this table reports, "
               "for each plausible discordance count $n_d=n_{10}+n_{01}$: the smallest "
               "$|n_{10}-n_{01}|$ that rejects $H_0$, the discordant-pair skew "
               "$\\delta_{\\text{skew}}=2|p_{10|d}-0.5|$ that reaches $80\\%$ power, and the "
               "corresponding rate-difference effect $\\delta_{\\text{pp}}=\\delta_{\\text{skew}}\\cdot n_d/N$ "
               "in percentage points. Cells with no rejection set are marked $\\infty$.}")
    out.append("    \\label{tab:prehoc-power}")
    out.append("    \\begin{tabular}{rrrrr}")
    out.append("        \\toprule")
    out.append("        $n_d$ & min $|n_{10}-n_{01}|$ to reject "
               "& $\\delta_{\\text{skew}}$ MDE & $\\delta_{\\text{pp}}$ MDE (pp) "
               "& $n_d/N$ \\\\")
    out.append("        \\midrule")
    for n_d in PREHOC_N_DISC_GRID:
        k_min = mcnemar_min_reject_k(n_d)
        skew  = mcnemar_exact_mde_skew(n_d)
        if k_min is None or skew is None:
            out.append(f"        {n_d:>3} & --- & $\\infty$ & $\\infty$ & {n_d/PREHOC_N_PAIRS:.2f} \\\\")
            continue
        pp = skew * (n_d / PREHOC_N_PAIRS) * 100
        out.append(f"        {n_d:>3} & {k_min:>3} & {skew:.2f} & {pp:.1f} & {n_d/PREHOC_N_PAIRS:.2f} \\\\")
    out.append("        \\bottomrule")
    out.append("    \\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Tables 4 & 5: per-variant statistical breakdowns (utility + safety)
#
# Each table mirrors tab_appendix_mde's per-cell statistics, but on the
# per-scenario pass indicator for a single variant rather than the joint-pass
# indicator. Splitting into two tables keeps the layout consistent with the
# joint-pass table (same 11 columns) and avoids the narrow-rendering issue a
# compact side-by-side layout produced.
# ---------------------------------------------------------------------------

def variant_cell_full_stats(records: dict[tuple, dict],
                            attack_type: str,
                            variant: str,
                            condition: str) -> dict | None:
    """Same shape as cell_stats() but on the per-scenario pass indicator
    for a single variant, comparing condition vs baseline."""
    pairs: dict[str, dict[str, bool]] = {}
    for (scen, cond), info in records.items():
        if info["type"] != attack_type or info["variant"] != variant:
            continue
        pairs.setdefault(scen, {})[cond] = info["passed"]

    n11 = n10 = n01 = n00 = 0
    base_v: list[bool] = []
    cond_v: list[bool] = []
    for scen, cs in pairs.items():
        if "baseline" not in cs or condition not in cs:
            continue
        b, c = cs["baseline"], cs[condition]
        base_v.append(b); cond_v.append(c)
        if b and c:           n11 += 1
        elif b and not c:     n10 += 1
        elif (not b) and c:   n01 += 1
        else:                 n00 += 1
    n = n00 + n01 + n10 + n11
    if n == 0:
        return None

    base_rate = sum(base_v) / n
    cond_rate = sum(cond_v) / n
    delta = cond_rate - base_rate
    ci_lo, ci_hi = bootstrap_delta_ci(base_v, cond_v)
    p   = mcnemar_exact_p(n10, n01)
    mde = mde_pp_exact(n, n10 + n01)
    noise = noise_fraction(n10, n01)
    odds = odds_ratio_corrected(n10, n01)
    return {
        "n": n, "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "base_rate": base_rate, "cond_rate": cond_rate,
        "delta": delta, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "p": p, "mde": mde, "noise": noise, "odds": odds,
    }


def _build_per_variant_table(per_model: dict[str, dict[tuple, dict]],
                             label: str,
                             tex_label: str,
                             variant_index: int,
                             caption_kind: str,
                             delta_symbol: str) -> str:
    """Emit a per-cell table on a single variant's pass indicator.
    `variant_index`: 0 for trusted (utility), 1 for untrusted (safety) per
    the FAMILIES tuple ordering."""
    out = []
    out.append("\\begin{small}")
    out.append("\\begin{xltabular}{\\linewidth}{@{\\extracolsep{\\fill}}lllrrrrrrrr@{}}")
    out.append(f"    \\caption{{Per-cell paired statistics on the {caption_kind} indicator. "
               f"All cells use $N=45$ paired scenarios. ${delta_symbol}$ is the change in pass "
               f"rate on the {caption_kind} variant under the test condition relative to "
               f"baseline; positive values indicate the condition produces more correct "
               f"outcomes on this variant. Row shading reflects the joint power analysis: "
               f"\\colorbox{{green!22}}{{green}} marks cells where $|{delta_symbol}|\\ge \\text{{MDE}}$ "
               f"at $80\\%$ power; \\colorbox{{yellow!28}}{{yellow}} marks cells that are "
               f"McNemar-significant at $\\alpha=0.05$ but with $|{delta_symbol}|<\\text{{MDE}}$; "
               f"unshaded rows are not significant or have no rejection region for the observed "
               f"discordance. Columns mirror Table~\\ref{{tab:appendix-mde}}: the effect "
               f"(${delta_symbol}$, 95\\% percentile bootstrap CI on ${delta_symbol}$, and the "
               f"Haldane-Anscombe-corrected McNemar odds ratio "
               f"$\\mathrm{{OR}}=(n_{{01}}+0.5)/(n_{{10}}+0.5)$) on the left; "
               f"discordance counts $n_{{10}}$ and $n_{{01}}$, the minimum detectable effect "
               f"at $80\\%$ power, the noise fraction $f_{{\\!N}}=\\min(n_{{10}},n_{{01}})/(n_{{10}}+n_{{01}})$, "
               f"and the exact McNemar $p$-value on the right.}}\\label{{{tex_label}}} \\\\")
    out.append("    \\toprule")
    out.append("                   &        &           "
               "& \\multicolumn{3}{c}{Effect} "
               "& \\multicolumn{5}{c}{Statistical testing} \\\\")
    out.append("    \\cmidrule(lr){4-6} \\cmidrule(lr){7-11}")
    out.append("    Model & Family & Condition "
               f"& ${delta_symbol}$ (pp) & 95\\% CI & OR "
               "& $n_{10}$ & $n_{01}$ & MDE@80\\% & $f_{\\!N}$ & $p$ \\\\")
    out.append("    \\midrule")
    out.append("    \\endfirsthead")
    out.append(f"    \\multicolumn{{11}}{{l}}{{\\textit{{Table~\\ref{{{tex_label}}} (continued)}}}} \\\\")
    out.append("    \\toprule")
    out.append("                   &        &           "
               "& \\multicolumn{3}{c}{Effect} "
               "& \\multicolumn{5}{c}{Statistical testing} \\\\")
    out.append("    \\cmidrule(lr){4-6} \\cmidrule(lr){7-11}")
    out.append("    Model & Family & Condition "
               f"& ${delta_symbol}$ (pp) & 95\\% CI & OR "
               "& $n_{10}$ & $n_{01}$ & MDE@80\\% & $f_{\\!N}$ & $p$ \\\\")
    out.append("    \\midrule")
    out.append("    \\endhead")
    out.append("    \\midrule \\multicolumn{11}{r}{\\textit{Continued on next page}} \\\\")
    out.append("    \\endfoot")
    out.append("    \\bottomrule")
    out.append("    \\endlastfoot")

    for slug in MODEL_ORDER:
        if slug not in per_model:
            continue
        records = per_model[slug]
        model_label = MODEL_LABEL.get(slug, slug)
        first_model_row = True
        for fam_key, fam_label, atype, tv, uv in FAMILIES:
            variant = (tv, uv)[variant_index]
            for cond in FAMILY_CONDITIONS[fam_key]:
                stats_cell = variant_cell_full_stats(records, atype, variant, cond)
                if stats_cell is None:
                    continue
                model_cell = model_label if first_model_row else ""
                first_model_row = False
                ci = f"[{fmt_pct(stats_cell['ci_lo'])}, {fmt_pct(stats_cell['ci_hi'])}]"
                mde_str = (fmt_pct_unsigned(stats_cell["mde"])
                           if stats_cell["mde"] is not None else "$\\infty$")
                noise_str = f"{stats_cell['noise']:.2f}" if stats_cell["noise"] is not None else "---"
                or_str = f"{stats_cell['odds']:.2f}" if stats_cell["odds"] is not None else "---"
                color = _row_color(stats_cell)
                out.append(
                    f"    {color}{model_cell:<14} & {fam_label:<10} & {COND_LABEL[cond]:<18} "
                    f"& {fmt_pct(stats_cell['delta']):>6} & {ci:>16} & {or_str:>5} "
                    f"& {stats_cell['n10']:>3} & {stats_cell['n01']:>3} "
                    f"& {mde_str:>8} & {noise_str:>5} & {fmt_p(stats_cell['p']):>8} \\\\"
                )
        out.append("    \\midrule")
    if out[-1] == "    \\midrule":
        out.pop()
    out.append("\\end{xltabular}")
    out.append("\\end{small}")
    return "\n".join(out) + "\n"


def build_utility_table(per_model: dict[str, dict[tuple, dict]]) -> str:
    return _build_per_variant_table(
        per_model,
        label="utility",
        tex_label="tab:appendix-utility",
        variant_index=0,  # trusted variant
        caption_kind="trusted (legitimate-request)",
        delta_symbol="\\Delta_{\\text{utility}}",
    )


def build_safety_table(per_model: dict[str, dict[tuple, dict]]) -> str:
    return _build_per_variant_table(
        per_model,
        label="safety",
        tex_label="tab:appendix-safety",
        variant_index=1,  # untrusted variant
        caption_kind="untrusted (attack)",
        delta_symbol="\\Delta_{\\text{safety}}",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    per_model: dict[str, dict[tuple, dict]] = {}
    for slug in MODEL_ORDER:
        recs = load_records(slug)
        if recs:
            per_model[slug] = recs
        else:
            print(f"  skipping {slug} — no eval_records.jsonl")

    if not per_model:
        raise SystemExit("No eval records found.")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    effects_path = TABLES_DIR / "tab_effects.tex"
    mde_path     = TABLES_DIR / "tab_appendix_mde.tex"
    prehoc_path  = TABLES_DIR / "tab_prehoc_power.tex"
    utility_path = TABLES_DIR / "tab_appendix_utility.tex"
    safety_path  = TABLES_DIR / "tab_appendix_safety.tex"

    effects_path.write_text(build_effects_table(per_model))
    mde_path.write_text(build_mde_table(per_model))
    prehoc_path.write_text(build_prehoc_power_table())
    utility_path.write_text(build_utility_table(per_model))
    safety_path.write_text(build_safety_table(per_model))

    # Remove the deprecated combined variant table so old appendix.tex
    # references will fail loudly rather than silently using stale stats.
    old = TABLES_DIR / "tab_appendix_variant.tex"
    if old.exists():
        old.unlink()

    print(f"  wrote {effects_path}")
    print(f"  wrote {mde_path}")
    print(f"  wrote {prehoc_path}")
    print(f"  wrote {utility_path}")
    print(f"  wrote {safety_path}")


if __name__ == "__main__":
    main()
