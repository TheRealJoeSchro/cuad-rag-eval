"""
Report generation: plain-text tables and summary written to reports/.
"""

from datetime import datetime
from pathlib import Path

from tabulate import tabulate

from eval.judge import JUDGE_RUBRIC
from eval.taxonomy import render_taxonomy_section


def write_report(
    cfg: dict,
    results: list[dict],
    taxonomy: dict,
    cross_chunk_count: int,
    total_gold_spans: int,
    match_type_counts: dict[str, int],
    report_name: str,
) -> str:
    """
    Write the eval report to reports/<report_name>.txt.

    Each item in results is a scored record with keys:
      contract_id, clause_type, gold_is_impossible,
      agent_is_absent, retrieval_recall, citation_iou, citation_hit,
      match_type, judge_correct, judge_justification,
      input_tokens, output_tokens, latency_s
    """
    out_dir = Path(cfg.get("reports_dir", "reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report_name}.txt"

    # Split by clause presence
    present = [r for r in results if not r["gold_is_impossible"]]
    absent = [r for r in results if r["gold_is_impossible"]]

    # --- Headline metrics ---
    fp_count = sum(1 for r in absent if not r["agent_is_absent"])
    fp_rate = fp_count / len(absent) if absent else 0.0
    tn_rate = 1.0 - fp_rate

    citation_hits = sum(1 for r in present if r["citation_hit"])
    citation_rate = citation_hits / len(present) if present else 0.0

    recall_hits = sum(1 for r in present if r["retrieval_recall"])
    recall_rate = recall_hits / len(present) if present else 0.0

    judge_present = [r for r in present if r["judge_correct"] is not None]
    judge_hits = sum(1 for r in judge_present if r["judge_correct"])
    judge_rate = judge_hits / len(judge_present) if judge_present else 0.0

    # --- Cost ---
    total_in = sum(r["input_tokens"] for r in results)
    total_out = sum(r["output_tokens"] for r in results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results) if results else 0
    avg_latency_present = sum(r["latency_s"] for r in present) / len(present) if present else 0
    avg_latency_absent = sum(r["latency_s"] for r in absent) / len(absent) if absent else 0
    avg_tok_present = (sum(r["input_tokens"] + r["output_tokens"] for r in present) / len(present)) if present else 0
    avg_tok_absent = (sum(r["input_tokens"] + r["output_tokens"] for r in absent) / len(absent)) if absent else 0

    # --- Per clause type ---
    clause_types = sorted(set(r["clause_type"] for r in results))
    per_clause = []
    for ct in clause_types:
        ct_results = [r for r in results if r["clause_type"] == ct]
        ct_present = [r for r in ct_results if not r["gold_is_impossible"]]
        ct_absent = [r for r in ct_results if r["gold_is_impossible"]]

        n_present = len(ct_present)
        n_absent = len(ct_absent)
        ct_recall = (sum(1 for r in ct_present if r["retrieval_recall"]) / n_present * 100) if n_present else None
        ct_citation = (sum(1 for r in ct_present if r["citation_hit"]) / n_present * 100) if n_present else None
        ct_fp = (sum(1 for r in ct_absent if not r["agent_is_absent"]) / n_absent * 100) if n_absent else None

        ct_judge_present = [r for r in ct_present if r["judge_correct"] is not None]
        ct_judge = (sum(1 for r in ct_judge_present if r["judge_correct"]) / len(ct_judge_present) * 100) if ct_judge_present else None

        per_clause.append([
            ct,
            n_present,
            n_absent,
            f"{ct_recall:.0f}%" if ct_recall is not None else "n/a",
            f"{ct_citation:.0f}%" if ct_citation is not None else "n/a",
            f"{ct_fp:.0f}%" if ct_fp is not None else "n/a",
            f"{ct_judge:.0f}%" if ct_judge is not None else "n/a",
        ])

    # --- Agent false-positive analysis ---
    # Among absent clauses where agent claimed present, show breakdown
    fp_by_clause = {}
    for r in absent:
        if not r["agent_is_absent"]:
            fp_by_clause[r["clause_type"]] = fp_by_clause.get(r["clause_type"], 0) + 1
    fp_table = sorted(fp_by_clause.items(), key=lambda x: -x[1])

    # --- Build report ---
    lines = []
    w = lines.append

    w("=" * 72)
    w("CUAD RAG EVAL REPORT")
    w("=" * 72)
    w(f"Generated:       {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"Agent model:     {cfg.get('llm_provider')}/{cfg.get('llm_model')}")
    w(f"Embedding:       {cfg.get('embedding_model')}")
    w(f"Retrieval k:     {cfg.get('retrieval_k')}")
    w(f"IoU threshold:   {cfg.get('iou_threshold')}")
    w(f"Judge model:     {cfg.get('llm_judge_provider')}/{cfg.get('llm_judge_model')}")

    n_contracts = len(set(r["contract_id"] for r in results))
    w(f"Contracts:       {n_contracts}")
    w(f"Total queries:   {len(results)} ({len(present)} present, {len(absent)} absent)")
    w("")

    w("-" * 72)
    w("HEADLINE METRICS")
    w("-" * 72)
    w(f"  False-Positive Rate:          {fp_rate*100:5.1f}%   "
      f"(agent claims present when gold says absent, {fp_count}/{len(absent)})")
    w(f"  Citation Grounding (IoU>={cfg.get('iou_threshold')}):  {citation_rate*100:5.1f}%   "
      f"(PRIMARY — on present clauses, {citation_hits}/{len(present)})")
    w(f"  Retrieval Recall@{cfg.get('retrieval_k')}:          {recall_rate*100:5.1f}%   "
      f"(retrieved chunk overlaps gold span, {recall_hits}/{len(present)})")
    w(f"  LLM-Judge Correctness:        {judge_rate*100:5.1f}%   "
      f"(SECONDARY — on present clauses, {judge_hits}/{len(judge_present)})")
    w("")

    w("-" * 72)
    w("BY CLAUSE PRESENCE")
    w("-" * 72)
    presence_table = [
        ["Retrieval recall@k",
         f"{recall_rate*100:.1f}%",
         "n/a"],
        ["Citation grounding",
         f"{citation_rate*100:.1f}%",
         "n/a"],
        ["False positive rate",
         "n/a",
         f"{fp_rate*100:.1f}%"],
        ["True negative rate",
         "n/a",
         f"{tn_rate*100:.1f}%"],
        ["LLM-judge correctness",
         f"{judge_rate*100:.1f}%",
         "n/a"],
        ["Avg latency (s)",
         f"{avg_latency_present:.2f}",
         f"{avg_latency_absent:.2f}"],
        ["Avg tokens (in+out)",
         f"{avg_tok_present:.0f}",
         f"{avg_tok_absent:.0f}"],
    ]
    w(tabulate(presence_table,
               headers=["", f"Present (N={len(present)})", f"Absent (N={len(absent)})"],
               tablefmt="simple"))
    w("")

    w("-" * 72)
    w("CITATION MATCH DIAGNOSTICS")
    w("-" * 72)
    w(f"  Exact match:            {match_type_counts.get('exact', 0)}")
    w(f"  Whitespace-rescued:     {match_type_counts.get('whitespace_rescued', 0)}")
    w(f"  Not found (true miss):  {match_type_counts.get('not_found', 0)}")
    agent_had_citation = sum(1 for r in present if not r["agent_is_absent"])
    agent_null = len(present) - agent_had_citation
    w(f"  Agent returned null:    {agent_null}  (agent said clause absent)")
    w("")

    w("-" * 72)
    w("PER CLAUSE TYPE")
    w("-" * 72)
    w(tabulate(per_clause,
               headers=["Clause Type", "N pres", "N abs", "Recall@k", "Citation", "FP Rate", "LLM-Judge"],
               tablefmt="simple"))
    w("")

    if fp_table:
        w("-" * 72)
        w("FALSE-POSITIVE BREAKDOWN (absent clauses where agent claimed present)")
        w("-" * 72)
        w(tabulate(fp_table, headers=["Clause Type", "FP Count"], tablefmt="simple"))
        w("")

    # Taxonomy section
    for line in render_taxonomy_section(taxonomy):
        w(line)

    w("-" * 72)
    w("STRUCTURAL NOTES")
    w("-" * 72)
    w(f"  {cross_chunk_count} of {total_gold_spans} gold spans ({100*cross_chunk_count/total_gold_spans:.1f}%) "
      f"cross chunk boundaries.")
    w("  These spans cannot achieve IoU=1.0 from a single chunk extraction.")
    w("  This is a structural ceiling of the chunking strategy, not a model error.")
    w("")

    w("-" * 72)
    w("COST SUMMARY")
    w("-" * 72)
    w(f"  Total queries:      {len(results)}")
    w(f"  Total tokens:       {total_in:,} in + {total_out:,} out")
    w(f"  Avg latency:        {avg_latency:.2f} s/query")
    w("")

    w("-" * 72)
    w("LLM-JUDGE METHOD NOTE")
    w("-" * 72)
    w(f"  Judge model: {cfg.get('llm_judge_provider')}/{cfg.get('llm_judge_model')}")
    w(f"  Agent model: {cfg.get('llm_provider')}/{cfg.get('llm_model')} (different model to avoid self-grading)")
    w("  Rubric prompt (verbatim):")
    for rubric_line in JUDGE_RUBRIC.split("\n"):
        w(f"    {rubric_line}")
    w("")
    w("=" * 72)

    report_text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report_text)

    return str(out_path)