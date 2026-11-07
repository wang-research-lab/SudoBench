#!/usr/bin/env python3
"""
OBSOLETE — kept for reference only.

This module references conditions (`signed_tags`, `unsigned_tags`) and a
variant (`untrusted_inline_spoof`) that the benchmark no longer evaluates.
Running it against current eval_summary.json files will produce stale
labels at best and KeyErrors at worst. Do not run until rewritten against
the current condition set: ["baseline", "tags", "capabilities", "both"].

report.py — Generate human-readable reports from eval_summary.json files.

Reads every results/<model>/eval_summary.json, writes a .txt report
to reports/<model>/report.txt, and prints a combined summary to stdout.

Usage
-----
  python report.py              # all models
  python report.py --model claude-sonnet-4-6
"""
import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
REPORTS_DIR = Path(__file__).parent / "reports"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _w(n: int = 100) -> str:
    return "─" * n


def _dw(n: int = 100) -> str:
    return "═" * n


def _mcnemar_row(label: str, s: dict, indent: int = 4) -> str:
    pad = " " * indent
    if s is None or s.get("chi2") is None:
        return f"{pad}{label:<36}  no discordant pairs"
    sig = "* " if s.get("significant_0.05") else "  "
    p01_str = f"p01={s['p01']:.3f} " if s.get("p01") is not None else ""
    return (
        f"{pad}{label:<36}  χ²={s['chi2']:>8.4f}  p={s['p']:.6f} {sig}"
        f" {p01_str}[n10={s['n10']} n01={s['n01']} n11={s['n11']} n00={s['n00']}]"
    )


def _mcnemar_block(block: dict, title: str, indent: int = 4) -> list[str]:
    lines = [f"{' ' * indent}{title}:"]
    lines.append(_mcnemar_row("overall", block["overall"], indent + 2))
    for k, v in block.get("by_type", {}).items():
        lines.append(_mcnemar_row(k, v, indent + 4))
    return lines


_PAIR_LABELS = {
    "unsigned_tags_vs_baseline":   "unsigned tags vs baseline  (do unsigned IFC tags help?)",
    "signed_tags_vs_baseline":     "signed tags vs baseline    (do signed IFC tags help?)",
    "signed_vs_unsigned_tags":     "signed vs unsigned tags    (does the signature earn its keep?)",
    "capabilities_vs_baseline":          "capabilities vs baseline         (do RBAC capabilities help?)",
    "full_vs_baseline":            "both vs baseline           (do composed primitives help?)",
    "full_vs_signed_tags":         "both vs signed tags        (do scopes add atop signed tags?)",
    "full_vs_capabilities":              "both vs capabilities             (do signed tags add atop capabilities?)",
}

_VARIANT_PAIR_LABELS = {
    "trusted_vs_untrusted":      "trusted vs untrusted        (baseline variant asymmetry)",
    "untrusted_vs_inline_spoof": "untrusted vs inline-spoof   (does inline tag forgery change outcomes?)",
}


def _mcnemar_section(mcn: dict) -> list[str]:
    lines = []
    var_cmp = mcn.get("variant_comparison", {})
    cond    = mcn.get("condition_effect", {})

    lines.append("  McNemar \u2014 variant comparisons")
    lines.append("  (H\u2080: call-rate equal across variants)")
    for key, by_cond in var_cmp.items():
        lines.append(f"    {_VARIANT_PAIR_LABELS.get(key, key)}:")
        for c in ["baseline", "unsigned_tags", "signed_tags", "capabilities", "both", "all_conditions"]:
            if c in by_cond:
                lines += _mcnemar_block(by_cond[c], c, indent=6)
    lines.append("")

    lines.append("  McNemar \u2014 condition effect")
    lines.append("  (H\u2080: changing exposed primitive(s) does not alter call-rate)")
    for key, block in cond.items():
        label = _PAIR_LABELS.get(key, key)
        lines.append(f"    {label}:")
        lines.append(_mcnemar_row("overall", block.get("overall"), indent=6))
        for k, v in block.get("by_variant", {}).items():
            lines.append(_mcnemar_row(f"variant={k}", v, indent=8))
        for k, v in block.get("by_type", {}).items():
            lines.append(_mcnemar_row(k, v, indent=8))
    return lines


# ---------------------------------------------------------------------------
# Per-model report
# ---------------------------------------------------------------------------

def build_report(summary: dict) -> str:
    W       = 110
    model   = summary["model"]
    run_at  = summary["run_at"]
    results = summary.get("results", [])
    mcn     = summary.get("mcnemar", {})

    lines = []
    lines.append(_dw(W))
    lines.append(f"  SOURCE AUTHORIZATION BENCHMARK  \u2014  {model}")
    lines.append(f"  run_at={run_at}  total={summary['total']}  "
                 f"passed={summary['passed']}  pass_rate={summary['pass_rate']:.1%}")
    lines.append(_dw(W))
    lines.append("")

    # ── attack type × authorization 2×2 table ───────────────────────────────
    #
    #  Rows  : IPI / Confused Deputy
    #  Cols  : Trusted (authorized) / Untrusted (unauthorized)
    #  These test fundamentally different failure modes:
    #    IPI×trusted      — delegated authority: agent should follow authenticated doc instructions
    #    IPI×untrusted    — classic IPI: agent must reject injected instructions
    #    CD×trusted       — authorized instruction following
    #    CD×untrusted     — classic confused deputy / privilege escalation
    #
    _COND_ORDER = {"baseline": 0, "unsigned_tags": 1, "signed_tags": 2, "capabilities": 3, "both": 4}
    _COND_LABEL = {"baseline": "baseline", "unsigned_tags": "unsigned_tags",
                   "signed_tags": "signed_tags", "capabilities": "capabilities", "both": "both"}
    _TYPE_LABEL = {"source_label_ipi":     "Source-label IPI",
                   "scope_check_ipi": "Scope-check IPI",
                   "confused_deputy":      "Confused deputy"}
    _VARIANT_ORDER  = {"trusted": 0, "untrusted": 1,
                       "in_scope": 2, "out_of_scope": 3}
    _VARIANT_HEADER = {"trusted":      ("Authorized",   "(trusted)"),
                       "untrusted":    ("Unauthorized", "(untrusted)"),
                       "in_scope":     ("In-scope",     "(act)"),
                       "out_of_scope": ("Out-of-scope", "(refuse)")}
    present_conds = sorted({r.get("condition", "baseline") for r in results},
                           key=lambda c: _COND_ORDER.get(c, 99))
    present_vars  = sorted({r.get("variant") for r in results if r.get("variant")},
                           key=lambda v: _VARIANT_ORDER.get(v, 99))

    if results:
        lines.append("  Attack type × Authorization  (pass rate = correct outcome):")
        col_w = 16
        h1 = "".join(f" {_VARIANT_HEADER.get(v, (v, ''))[0]:>{col_w}} " for v in present_vars)
        h2 = "".join(f" {_VARIANT_HEADER.get(v, ('', v))[1]:>{col_w}} " for v in present_vars)
        lines.append(f"  {'':34}{h1}")
        lines.append(f"  {'':34}{h2}")
        for cond in present_conds:
            lines.append(f"  {_COND_LABEL.get(cond, cond)}:")
            for atype in ["source_label_ipi", "scope_check_ipi", "confused_deputy"]:
                row_cells = []
                for variant in present_vars:
                    cell = [r for r in results
                            if r.get("condition") == cond
                            and r["type"] == atype
                            and r["variant"] == variant]
                    if cell:
                        p = sum(1 for r in cell if r["passed"])
                        row_cells.append(f"{p}/{len(cell)} ({p/len(cell):.0%})")
                    else:
                        row_cells.append("—")
                tlabel = _TYPE_LABEL.get(atype, atype)
                cells_str = "".join(f" {c:>{col_w}} " for c in row_cells)
                lines.append(f"    {tlabel:<32}{cells_str}")
        lines.append("")

    # ── pass rate by condition ───────────────────────────────────────────────
    by_cond = summary.get("by_condition", {})
    if by_cond:
        lines.append("  Pass rate by condition (aggregate):")
        for cond in present_conds:
            if cond not in by_cond:
                continue
            stats = by_cond[cond]
            lines.append(f"    {_COND_LABEL.get(cond, cond):<18}  {stats['passed']}/{stats['total']}  "
                         f"({stats['pass_rate']:.1%})")
        lines.append("")

    # ── pass rate by boundary type ───────────────────────────────────────────
    boundary_totals: dict[str, list] = {}
    for r in results:
        boundary_totals.setdefault(r["boundary_type"], []).append(r["passed"])
    if boundary_totals:
        lines.append("  Pass rate by boundary type (all conditions pooled):")
        for bt in sorted(boundary_totals):
            ps = boundary_totals[bt]
            lines.append(f"    {bt:<24}  {sum(ps)}/{len(ps)}  ({sum(ps)/len(ps):.1%})")
        lines.append("")

    # ── McNemar ──────────────────────────────────────────────────────────────
    if mcn:
        lines.append(_w(W))
        lines += _mcnemar_section(mcn)
        lines.append("")

    # ── per-row results table ────────────────────────────────────────────────
    lines.append(_w(W))
    lines.append(f"  {'Scenario':<36} {'Type':<22} {'Boundary':<22} "
                 f"{'Variant':<16} {'Cond':<12} {'Expected':<14} {'Actual':<14} Result")
    lines.append(f"  {_w(W)}")
    _COND_ABBREV = {"baseline": "base", "unsigned_tags": "unsig_tg",
                    "signed_tags": "sig_tags", "capabilities": "capabilities", "both": "both"}
    _key = lambda x: (_COND_ORDER.get(x.get("condition") or "baseline", 99),
                      x["type"],
                      _VARIANT_ORDER.get(x.get("variant") or "", 99),
                      x["scenario"])
    for r in sorted(results, key=_key):
        mark   = "PASS \u2713" if r["passed"] else "FAIL \u2717"
        cond   = r.get("condition") or "baseline"
        clabel = _COND_ABBREV.get(cond, cond[:10])
        lines.append(
            f"  {r['scenario']:<36} {r['type']:<22} {r['boundary_type']:<22} "
            f"{r['variant']:<16} {clabel:<12} {r['expected']:<14} {r['actual']:<14} {mark}"
        )
    lines.append(f"  {_w(W)}")
    lines.append("")

    # ── evidence (detail strings) ────────────────────────────────────────────
    lines.append("  Evidence (model output per scenario):")
    lines.append(f"  {_w(W)}")
    for r in sorted(results, key=_key):
        status = "PASS" if r["passed"] else "FAIL"
        cond   = r.get("condition") or "baseline"
        lines.append(f"  [{r['scenario']} | {r['variant']} | {cond} | {status}]")
        detail = (r.get("detail") or "").strip()
        if detail:
            lines.append(f"    {detail}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate benchmark reports from eval summaries")
    p.add_argument("--model", default=None,
                   help="Model slug to report on (default: all models in results/)")
    return p.parse_args()


def main():
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_dirs = (
        [RESULTS_DIR / args.model] if args.model
        else sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
    )

    if not model_dirs:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    for model_dir in model_dirs:
        summary_path = model_dir / "eval_summary.json"
        if not summary_path.exists():
            print(f"  skipping {model_dir.name} — no eval_summary.json", file=sys.stderr)
            continue

        summary  = json.loads(summary_path.read_text())
        report   = build_report(summary)

        out_dir  = REPORTS_DIR / model_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "report.txt"
        out_path.write_text(report)

        print(f"  Saved \u2192 {out_path}\n")


if __name__ == "__main__":
    main()
