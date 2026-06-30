"""
Comparison report: side-by-side metrics across named runs.

Computes headline metrics, taxonomy deltas, cost from actual token counts
with pinned prices, and a mechanical hypothesis verdict.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tabulate import tabulate

from eval.taxonomy import build_taxonomy, CATEGORY_SUCCESS, CATEGORY_WRONG_RET, \
    CATEGORY_WRONG_EXTRACT, CATEGORY_MISSED_PRESENT, CATEGORY_HALLUCINATED, CATEGORY_TRUE_NEG


# Pinned prices: (input $/MTok, output $/MTok)
MODEL_PRICES = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}


def _load_scored(processed_dir: str, run_name: str) -> list[dict]:
    path = Path(processed_dir) / f"scored_results_{run_name}.jsonl"
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def _detect_model(run_name: str) -> str:
    """Infer agent model from run name convention."""
    if "sonnet" in run_name.lower():
        return "claude-sonnet-4-6"
    return "claude-haiku-4-5-20251001"


def _compute_metrics(results: list[dict], iou_threshold: float) -> dict:
    present = [r for r in results if not r["gold_is_impossible"]]
    absent = [r for r in results if r["gold_is_impossible"]]

    fp_count = sum(1 for r in absent if not r["agent_is_absent"])
    fp_rate = fp_count / len(absent) if absent else 0.0

    citation_hits = sum(1 for r in present if r["citation_hit"])
    citation_rate = citation_hits / len(present) if present else 0.0

    recall_hits = sum(1 for r in present if r["retrieval_recall"])
    recall_rate = recall_hits / len(present) if present else 0.0

    judge_present = [r for r in present if r["judge_correct"] is not None]
    judge_hits = sum(1 for r in judge_present if r["judge_correct"])
    judge_rate = judge_hits / len(judge_present) if judge_present else 0.0

    total_in = sum(r["input_tokens"] for r in results)
    total_out = sum(r["output_tokens"] for r in results)

    taxonomy = build_taxonomy(results)
    tax_counts = {}
    for cat in [CATEGORY_SUCCESS, CATEGORY_WRONG_RET, CATEGORY_WRONG_EXTRACT,
                CATEGORY_MISSED_PRESENT, CATEGORY_HALLUCINATED, CATEGORY_TRUE_NEG]:
        tax_counts[cat] = len(taxonomy.get(cat, []))

    return {
        "n_present": len(present),
        "n_absent": len(absent),
        "n_total": len(results),
        "fp_rate": fp_rate,
        "fp_count": fp_count,
        "citation_rate": citation_rate,
        "citation_hits": citation_hits,
        "recall_rate": recall_rate,
        "recall_hits": recall_hits,
        "judge_rate": judge_rate,
        "judge_hits": judge_hits,
        "judge_n": len(judge_present),
        "total_in": total_in,
        "total_out": total_out,
        "taxonomy": tax_counts,
    }


def _compute_cost(total_in: int, total_out: int, model: str) -> float:
    """Cost in dollars from actual token counts and pinned prices."""
    price_in, price_out = MODEL_PRICES.get(model, (1.0, 5.0))
    return (total_in * price_in + total_out * price_out) / 1_000_000


def generate_comparison(cfg: dict, run_names: list[str]) -> None:
    processed_dir = cfg.get("processed_dir", "data/processed")
    out_dir = Path(cfg.get("reports_dir", "reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    iou_threshold = cfg.get("iou_threshold", 0.5)

    # Load and compute metrics for each run
    runs = {}
    for name in run_names:
        results = _load_scored(processed_dir, name)
        metrics = _compute_metrics(results, iou_threshold)
        model = _detect_model(name)
        metrics["model"] = model
        metrics["cost"] = _compute_cost(metrics["total_in"], metrics["total_out"], model)
        runs[name] = metrics

    # Detect retrieval_k from run name
    def _k_from_name(name: str) -> int:
        if "k10" in name:
            return 10
        if "k5" in name:
            return 5
        return cfg.get("retrieval_k", 5)

    # Build report
    lines = []
    w = lines.append

    w("=" * 80)
    w("CUAD RAG EVAL — COMPARISON REPORT")
    w("=" * 80)
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"Runs:      {', '.join(run_names)}")
    w("")

    # --- Headline metrics table ---
    w("-" * 80)
    w("HEADLINE METRICS")
    w("-" * 80)

    metric_rows = []
    headers = ["Metric"] + run_names

    baseline = runs[run_names[0]]

    def _delta(val, base, fmt=".1f", pct=True):
        d = val - base
        sign = "+" if d > 0 else ""
        suffix = "pp" if pct else ""
        return f" ({sign}{d*100 if pct else d:{fmt}}{suffix})"

    for label, key, is_pct in [
        ("Citation Grounding", "citation_rate", True),
        ("Retrieval Recall@k", "recall_rate", True),
        ("False-Positive Rate", "fp_rate", True),
        ("LLM-Judge Correct", "judge_rate", True),
    ]:
        row = [label]
        for i, name in enumerate(run_names):
            m = runs[name]
            val_str = f"{m[key]*100:.1f}%"
            if i > 0:
                val_str += _delta(m[key], baseline[key])
            k = _k_from_name(name)
            if key == "recall_rate":
                row.append(f"{val_str} @{k}")
            else:
                row.append(val_str)
        metric_rows.append(row)

    w(tabulate(metric_rows, headers=headers, tablefmt="simple"))
    w("")

    # --- Taxonomy comparison ---
    w("-" * 80)
    w("TAXONOMY COMPARISON (present clauses)")
    w("-" * 80)

    present_cats = [CATEGORY_SUCCESS, CATEGORY_WRONG_RET, CATEGORY_WRONG_EXTRACT, CATEGORY_MISSED_PRESENT]
    tax_rows = []
    for cat in present_cats:
        row = [cat]
        for i, name in enumerate(run_names):
            m = runs[name]
            n = m["taxonomy"][cat]
            pct = 100 * n / m["n_present"] if m["n_present"] else 0
            cell = f"{n} ({pct:.1f}%)"
            if i > 0:
                base_n = baseline["taxonomy"][cat]
                delta = n - base_n
                sign = "+" if delta > 0 else ""
                cell += f"  [{sign}{delta}]"
            row.append(cell)
        tax_rows.append(row)

    w(tabulate(tax_rows, headers=headers, tablefmt="simple"))
    w("")

    # Absent clauses
    w("-" * 80)
    w("TAXONOMY COMPARISON (absent clauses)")
    w("-" * 80)
    absent_cats = [CATEGORY_TRUE_NEG, CATEGORY_HALLUCINATED]
    abs_rows = []
    for cat in absent_cats:
        row = [cat]
        for i, name in enumerate(run_names):
            m = runs[name]
            n = m["taxonomy"][cat]
            pct = 100 * n / m["n_absent"] if m["n_absent"] else 0
            cell = f"{n} ({pct:.1f}%)"
            if i > 0:
                base_n = baseline["taxonomy"][cat]
                delta = n - base_n
                sign = "+" if delta > 0 else ""
                cell += f"  [{sign}{delta}]"
            row.append(cell)
        abs_rows.append(row)

    w(tabulate(abs_rows, headers=headers, tablefmt="simple"))
    w("")

    # --- Cost comparison ---
    w("-" * 80)
    w("COST COMPARISON (actual tokens x pinned prices)")
    w("-" * 80)
    w("  Pinned prices (per million tokens):")
    for model, (p_in, p_out) in MODEL_PRICES.items():
        w(f"    {model}: ${p_in:.0f} in / ${p_out:.0f} out")
    w("")

    cost_rows = []
    cost_headers = ["Run", "Model", "Tokens In", "Tokens Out", "Cost", "vs Baseline"]
    for i, name in enumerate(run_names):
        m = runs[name]
        cost_str = f"${m['cost']:.4f}"
        if i == 0:
            vs = "—"
        else:
            ratio = m["cost"] / baseline["cost"] if baseline["cost"] > 0 else 0
            vs = f"{ratio:.1f}x"
        cost_rows.append([
            name, m["model"],
            f"{m['total_in']:,}", f"{m['total_out']:,}",
            cost_str, vs,
        ])
    w(tabulate(cost_rows, headers=cost_headers, tablefmt="simple"))
    w("")

    # --- Retrieval ceiling diagnosis (for runs with different k) ---
    k_values = {name: _k_from_name(name) for name in run_names}
    if len(set(k_values.values())) > 1:
        w("-" * 80)
        w("RETRIEVAL CEILING DIAGNOSIS")
        w("-" * 80)
        # Find the run pair where k differs
        for name in run_names[1:]:
            if k_values[name] != k_values[run_names[0]]:
                m = runs[name]
                b = baseline
                recall_delta = m["recall_rate"] - b["recall_rate"]
                wr_delta = m["taxonomy"][CATEGORY_WRONG_RET] - b["taxonomy"][CATEGORY_WRONG_RET]
                w(f"  {run_names[0]} (k={k_values[run_names[0]]}): recall={b['recall_rate']*100:.1f}%  "
                  f"WRONG_RETRIEVED={b['taxonomy'][CATEGORY_WRONG_RET]}")
                w(f"  {name} (k={k_values[name]}): recall={m['recall_rate']*100:.1f}%  "
                  f"WRONG_RETRIEVED={m['taxonomy'][CATEGORY_WRONG_RET]}")
                w(f"  Recall delta: {recall_delta*100:+.1f}pp  |  WRONG_RETRIEVED delta: {wr_delta:+d}")
                w("")
                if recall_delta > 0.05:
                    w(f"  Verdict: k={k_values[name]} lifted recall meaningfully "
                      f"({recall_delta*100:+.1f}pp) and shrank WRONG_RETRIEVED ({wr_delta:+d}).")
                    w(f"  The retrieval ceiling at k={k_values[run_names[0]]} was limiting performance.")
                elif recall_delta > 0:
                    w(f"  Verdict: k={k_values[name]} gave marginal recall lift "
                      f"({recall_delta*100:+.1f}pp).")
                    w(f"  The ceiling is primarily retrieval quality (embedding/chunking), not k.")
                else:
                    w(f"  Verdict: k={k_values[name]} did not improve recall. "
                      f"The ceiling is retrieval quality (embedding/chunking), not k.")
        w("")

    # --- Hypothesis verdict ---
    w("-" * 80)
    w("HYPOTHESIS VERDICT")
    w("-" * 80)

    for name in run_names[1:]:
        m = runs[name]
        w(f"\n  {name} vs {run_names[0]}:")

        # Citation grounding
        cit_delta = m["citation_rate"] - baseline["citation_rate"]
        w(f"    Citation grounding: {cit_delta*100:+.1f}pp")

        # FP rate
        fp_delta = m["fp_rate"] - baseline["fp_rate"]
        w(f"    False-positive rate: {fp_delta*100:+.1f}pp")

        # WRONG_EXTRACTION delta (agent quality)
        we_delta = m["taxonomy"][CATEGORY_WRONG_EXTRACT] - baseline["taxonomy"][CATEGORY_WRONG_EXTRACT]
        w(f"    WRONG_EXTRACTION: {we_delta:+d}")

        # HALLUCINATED delta (agent quality)
        hall_delta = m["taxonomy"][CATEGORY_HALLUCINATED] - baseline["taxonomy"][CATEGORY_HALLUCINATED]
        w(f"    HALLUCINATED: {hall_delta:+d}")

        # WRONG_RETRIEVED delta (retrieval quality — should be flat for model-only changes)
        wr_delta = m["taxonomy"][CATEGORY_WRONG_RET] - baseline["taxonomy"][CATEGORY_WRONG_RET]
        w(f"    WRONG_RETRIEVED: {wr_delta:+d} (expect ~0 for model-only changes)")

        # Cost
        if baseline["cost"] > 0:
            ratio = m["cost"] / baseline["cost"]
            w(f"    Cost: {ratio:.1f}x baseline")

        # Mechanical verdict
        improved = (cit_delta > 0.02) or (fp_delta < -0.02)
        regressed = (cit_delta < -0.02) or (fp_delta > 0.02)
        if improved and not regressed:
            w("    -> IMPROVED over baseline")
        elif regressed and not improved:
            w("    -> REGRESSED from baseline")
        elif improved and regressed:
            w("    -> MIXED: improved on some metrics, regressed on others")
        else:
            w("    -> FLAT: no meaningful change from baseline")

    w("")
    w("=" * 80)

    report_text = "\n".join(lines)
    out_path = out_dir / "comparison.txt"
    with open(out_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nComparison report written: {out_path}")
