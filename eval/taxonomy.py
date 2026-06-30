"""
Failure taxonomy: classify every non-success outcome into one of four categories.

Categories (present clauses):
  WRONG_RETRIEVED    — no retrieved chunk overlapped the gold span; agent never saw
                       the clause. Includes cases where agent still gave an answer
                       from wrong chunks (citing real but irrelevant text) and cases
                       where retrieval miss caused agent to abstain.
  WRONG_EXTRACTION   — retrieval hit (right chunk was retrieved), but citation grounding
                       failed. Splits into: agent gave no verbatim citation, or agent
                       cited real text that was sub-threshold IoU against gold.
  MISSED_BUT_PRESENT — gold clause is present, agent said absent.
                       Sub-split: retrieval also missed vs. retrieval hit but agent
                       still abstained.

Category (absent clauses):
  HALLUCINATED       — gold says clause is absent; agent claimed it was present.

Every (contract, clause_type) pair lands in exactly one bucket:
  present → SUCCESS | WRONG_RETRIEVED | WRONG_EXTRACTION | MISSED_BUT_PRESENT
  absent  → TRUE_NEGATIVE | HALLUCINATED
"""

from collections import defaultdict
from pathlib import Path
from tabulate import tabulate


CATEGORY_SUCCESS        = "SUCCESS"
CATEGORY_WRONG_RET      = "WRONG_RETRIEVED"
CATEGORY_WRONG_EXTRACT  = "WRONG_EXTRACTION"
CATEGORY_MISSED_PRESENT = "MISSED_BUT_PRESENT"
CATEGORY_HALLUCINATED   = "HALLUCINATED"
CATEGORY_TRUE_NEG       = "TRUE_NEGATIVE"


def classify(result: dict) -> tuple[str, str]:
    """
    Return (category, sub_note) for a scored result.

    sub_note gives a one-line detail useful for the breakdown table.
    """
    gold_absent  = result["gold_is_impossible"]
    agent_absent = result["agent_is_absent"]
    recall       = result["retrieval_recall"]
    citation_hit = result["citation_hit"]
    match_type   = result["match_type"]  # "exact" | "whitespace_rescued" | "not_found" | "n/a"

    # --- Absent clauses ---
    if gold_absent:
        if agent_absent:
            return CATEGORY_TRUE_NEG, "correctly abstained"
        else:
            return CATEGORY_HALLUCINATED, "claimed present"

    # --- Present clauses ---
    if citation_hit:
        return CATEGORY_SUCCESS, f"IoU≥threshold ({match_type})"

    if agent_absent:
        if not recall:
            return CATEGORY_MISSED_PRESENT, "retrieval also missed"
        else:
            return CATEGORY_MISSED_PRESENT, "right chunk retrieved; agent abstained"

    # Agent gave an answer (not absent), but citation failed
    if not recall:
        if match_type == "exact":
            return CATEGORY_WRONG_RET, "cited real text from wrong location (IoU=0)"
        else:
            return CATEGORY_WRONG_RET, "gold chunk not retrieved; no usable citation"
    else:
        # Retrieval hit but citation didn't pass
        if match_type == "n/a":
            return CATEGORY_WRONG_EXTRACT, "right chunk retrieved; no verbatim citation provided"
        else:
            return CATEGORY_WRONG_EXTRACT, f"right chunk retrieved; citation sub-threshold IoU ({result['citation_iou']:.3f})"


def build_taxonomy(results: list[dict]) -> dict:
    """
    Classify all results. Returns a taxonomy dict with counts and per-clause breakdowns.
    """
    categories = defaultdict(list)
    for r in results:
        cat, sub = classify(r)
        categories[cat].append({**r, "_sub_note": sub})

    return dict(categories)


def render_taxonomy_section(taxonomy: dict) -> list[str]:
    """
    Return lines for the taxonomy section of the report.
    """
    lines = []
    w = lines.append

    present_total = sum(
        len(taxonomy.get(c, []))
        for c in [CATEGORY_SUCCESS, CATEGORY_WRONG_RET,
                  CATEGORY_WRONG_EXTRACT, CATEGORY_MISSED_PRESENT]
    )
    absent_total = sum(
        len(taxonomy.get(c, []))
        for c in [CATEGORY_TRUE_NEG, CATEGORY_HALLUCINATED]
    )

    w("-" * 72)
    w("FAILURE TAXONOMY")
    w("-" * 72)
    w("")
    w("PRESENT CLAUSES")
    w(f"  (N={present_total} present-clause pairs)")
    w("")

    success_n   = len(taxonomy.get(CATEGORY_SUCCESS, []))
    wrong_ret_n = len(taxonomy.get(CATEGORY_WRONG_RET, []))
    wrong_ext_n = len(taxonomy.get(CATEGORY_WRONG_EXTRACT, []))
    missed_n    = len(taxonomy.get(CATEGORY_MISSED_PRESENT, []))

    pct = lambda n: f"{100*n/present_total:.1f}%" if present_total else "n/a"

    summary_rows = [
        ["SUCCESS",             success_n,   pct(success_n),
         "citation IoU >= threshold"],
        ["WRONG_RETRIEVED",     wrong_ret_n, pct(wrong_ret_n),
         "gold chunk never retrieved; agent worked from wrong context"],
        ["WRONG_EXTRACTION",    wrong_ext_n, pct(wrong_ext_n),
         "right chunk retrieved; agent failed to extract correct span"],
        ["MISSED_BUT_PRESENT",  missed_n,    pct(missed_n),
         "agent said clause absent; gold says it exists"],
    ]
    w(tabulate(summary_rows,
               headers=["Category", "N", "%", "Description"],
               tablefmt="simple"))
    w("")

    # Sub-breakdowns for each failure category
    for cat, label in [
        (CATEGORY_WRONG_RET,     "WRONG_RETRIEVED sub-breakdown"),
        (CATEGORY_WRONG_EXTRACT, "WRONG_EXTRACTION sub-breakdown"),
        (CATEGORY_MISSED_PRESENT,"MISSED_BUT_PRESENT sub-breakdown"),
    ]:
        items = taxonomy.get(cat, [])
        if not items:
            continue
        sub_counts = defaultdict(int)
        for item in items:
            sub_counts[item["_sub_note"]] += 1
        w(f"  {label}:")
        for sub, count in sorted(sub_counts.items(), key=lambda x: -x[1]):
            w(f"    {count:3d}  {sub}")
        w("")

    # Wrong retrieved: per clause type
    wr_items = taxonomy.get(CATEGORY_WRONG_RET, [])
    if wr_items:
        ct_counts = defaultdict(int)
        for item in wr_items:
            ct_counts[item["clause_type"]] += 1
        w("  WRONG_RETRIEVED by clause type (top 10):")
        rows = sorted(ct_counts.items(), key=lambda x: -x[1])[:10]
        w(tabulate(rows, headers=["Clause Type", "N"], tablefmt="simple"))
        w("")

    # Wrong extraction: per clause type
    we_items = taxonomy.get(CATEGORY_WRONG_EXTRACT, [])
    if we_items:
        ct_counts = defaultdict(int)
        for item in we_items:
            ct_counts[item["clause_type"]] += 1
        w("  WRONG_EXTRACTION by clause type (top 10):")
        rows = sorted(ct_counts.items(), key=lambda x: -x[1])[:10]
        w(tabulate(rows, headers=["Clause Type", "N"], tablefmt="simple"))
        w("")

    w("")
    w("ABSENT CLAUSES")
    w(f"  (N={absent_total} absent-clause pairs)")
    w("")

    halluc_n = len(taxonomy.get(CATEGORY_HALLUCINATED, []))
    tn_n     = len(taxonomy.get(CATEGORY_TRUE_NEG, []))
    apct = lambda n: f"{100*n/absent_total:.1f}%" if absent_total else "n/a"

    absent_rows = [
        ["TRUE_NEGATIVE",  tn_n,     apct(tn_n),
         "agent correctly said clause absent"],
        ["HALLUCINATED",   halluc_n, apct(halluc_n),
         "agent claimed present; gold says absent (false positive)"],
    ]
    w(tabulate(absent_rows,
               headers=["Category", "N", "%", "Description"],
               tablefmt="simple"))
    w("")

    # Hallucination by clause type
    hall_items = taxonomy.get(CATEGORY_HALLUCINATED, [])
    if hall_items:
        ct_counts = defaultdict(int)
        for item in hall_items:
            ct_counts[item["clause_type"]] += 1
        w("  HALLUCINATED by clause type (top 15):")
        rows = sorted(ct_counts.items(), key=lambda x: -x[1])[:15]
        w(tabulate(rows, headers=["Clause Type", "FP Count"], tablefmt="simple"))
        w("")

    return lines
