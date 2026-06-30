"""
Eval harness: score agent outputs against CUAD gold labels.

Orchestrates scoring, LLM-judge calls, and report generation.
"""

import json
from pathlib import Path
from tqdm import tqdm

from eval.scoring import (
    load_contracts,
    load_chunks_by_id,
    load_chunks_by_contract,
    locate_citation,
    max_iou_vs_gold,
    retrieval_recall,
    count_cross_chunk_spans,
)
from eval.judge import judge_answer
from eval.report import write_report
from eval.taxonomy import build_taxonomy


def _agent_outputs_path(processed_dir: str, run_name: str | None) -> Path:
    if run_name:
        return Path(processed_dir) / f"agent_outputs_{run_name}.jsonl"
    return Path(processed_dir) / "agent_outputs.jsonl"


def _scored_results_path(processed_dir: str, run_name: str | None) -> Path:
    if run_name:
        return Path(processed_dir) / f"scored_results_{run_name}.jsonl"
    return Path(processed_dir) / "scored_results.jsonl"


def _load_agent_outputs(processed_dir: str, run_name: str | None) -> list[dict]:
    path = _agent_outputs_path(processed_dir, run_name)
    outputs = []
    with open(path) as f:
        for line in f:
            outputs.append(json.loads(line))
    return outputs


def run_eval(
    cfg: dict,
    report_name: str = "report",
    run_name: str | None = None,
    skip_judge: bool = False,
) -> None:
    processed_dir = cfg.get("processed_dir", "data/processed")
    iou_threshold = cfg.get("iou_threshold", 0.5)

    print("Loading data...")
    contracts = load_contracts(processed_dir)
    outputs = _load_agent_outputs(processed_dir, run_name)
    chunks_by_id = load_chunks_by_id(processed_dir)
    chunks_by_contract = load_chunks_by_contract(processed_dir)

    # Build gold label lookup: (contract_id, clause_type) -> gold_label
    gold_lookup: dict[tuple[str, str], dict] = {}
    for cid, crec in contracts.items():
        for label in crec["gold_labels"]:
            gold_lookup[(cid, label["clause_type"])] = label

    # Count cross-chunk gold spans across all contracts
    total_cross = 0
    total_spans = 0
    for cid, crec in contracts.items():
        cchunks = chunks_by_contract.get(cid, [])
        for label in crec["gold_labels"]:
            if label["answers"]:
                cc, tt = count_cross_chunk_spans(label["answers"], cchunks)
                total_cross += cc
                total_spans += tt

    # Load existing judge scores to preserve them when --skip-judge
    scored_path = _scored_results_path(processed_dir, run_name)
    existing_judge: dict[tuple[str, str], tuple[bool | None, str]] = {}
    if skip_judge and scored_path.exists():
        with open(scored_path) as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["contract_id"], rec["clause_type"])
                existing_judge[key] = (rec.get("judge_correct"), rec.get("judge_justification", ""))

    # Score each agent output
    results = []
    match_type_counts = {"exact": 0, "whitespace_rescued": 0, "not_found": 0}

    # Collect judge work items (only for present clauses where agent gave an answer)
    judge_items = []

    print(f"Scoring {len(outputs)} outputs...")
    for out in tqdm(outputs, desc="Scoring"):
        cid = out["contract_id"]
        ct = out["clause_type"]
        gold = gold_lookup.get((cid, ct))

        if gold is None:
            continue  # skip outputs without matching gold labels

        gold_impossible = gold["is_impossible"]
        gold_answers = gold["answers"]
        context = contracts[cid]["context"]

        # Retrieval recall (only meaningful for present clauses)
        recall = False
        if not gold_impossible:
            recall = retrieval_recall(
                out["retrieved_chunk_ids"],
                gold_answers,
                chunks_by_id,
            )

        # Citation grounding
        citation_iou = 0.0
        citation_hit = False
        match_type = "n/a"

        if not gold_impossible and not out["is_absent"] and out.get("supporting_clause_text"):
            # Look up chunk span for tie-breaking
            source_chunk = chunks_by_id.get(out.get("source_citation", ""))
            chunk_start = source_chunk["char_start"] if source_chunk else None
            chunk_end = source_chunk["char_end"] if source_chunk else None

            pred_span, match_type = locate_citation(
                out["supporting_clause_text"],
                context,
                chunk_start,
                chunk_end,
            )

            if pred_span is not None:
                citation_iou = max_iou_vs_gold(pred_span, gold_answers)
                citation_hit = citation_iou >= iou_threshold

            match_type_counts[match_type] = match_type_counts.get(match_type, 0) + 1

        prior_judge, prior_just = existing_judge.get((cid, ct), (None, ""))
        result = {
            "contract_id": cid,
            "clause_type": ct,
            "gold_is_impossible": gold_impossible,
            "agent_is_absent": out["is_absent"],
            "retrieval_recall": recall,
            "citation_iou": round(citation_iou, 4),
            "citation_hit": citation_hit,
            "match_type": match_type,
            "judge_correct": prior_judge,
            "judge_justification": prior_just,
            "input_tokens": out["input_tokens"],
            "output_tokens": out["output_tokens"],
            "latency_s": out["latency_s"],
        }
        results.append(result)

        # Queue for LLM judge if present clause and agent gave an answer
        if not gold_impossible and not out["is_absent"] and out.get("answer"):
            judge_items.append((len(results) - 1, out["answer"], gold_answers))

    # Run LLM-as-judge (secondary metric)
    judge_provider = cfg.get("llm_judge_provider", "anthropic")
    judge_model = cfg.get("llm_judge_model", "claude-sonnet-4-6")

    if judge_items and not skip_judge:
        print(f"Running LLM-as-judge ({judge_model}) on {len(judge_items)} present-clause answers...")
        for idx, answer, g_answers in tqdm(judge_items, desc="Judging"):
            correct, justification = judge_answer(
                agent_answer=answer,
                gold_answers=g_answers,
                provider=judge_provider,
                model=judge_model,
            )
            results[idx]["judge_correct"] = correct
            results[idx]["judge_justification"] = justification
    elif skip_judge:
        print("Skipping LLM-as-judge (--skip-judge set).")

    # Write scored results
    with open(scored_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Scored results: {scored_path}")

    # Build taxonomy and generate report
    taxonomy = build_taxonomy(results)
    report_path = write_report(
        cfg=cfg,
        results=results,
        taxonomy=taxonomy,
        cross_chunk_count=total_cross,
        total_gold_spans=total_spans,
        match_type_counts=match_type_counts,
        report_name=report_name,
    )
    print(f"Report written: {report_path}")


def run_taxonomy(cfg: dict, report_name: str = "taxonomy") -> None:
    """Re-run taxonomy + report from existing scored_results.jsonl (no LLM calls)."""
    processed_dir = cfg.get("processed_dir", "data/processed")
    scored_path = Path(processed_dir) / "scored_results.jsonl"
    if not scored_path.exists():
        print("No scored_results.jsonl found. Run `python cli.py eval` first.")
        return

    results = []
    with open(scored_path) as f:
        for line in f:
            results.append(json.loads(line))

    # Reload cross-chunk count
    contracts = load_contracts(processed_dir)
    chunks_by_contract = load_chunks_by_contract(processed_dir)
    total_cross, total_spans = 0, 0
    for cid, crec in contracts.items():
        cchunks = chunks_by_contract.get(cid, [])
        for label in crec["gold_labels"]:
            if label["answers"]:
                cc, tt = count_cross_chunk_spans(label["answers"], cchunks)
                total_cross += cc
                total_spans += tt

    match_type_counts = {"exact": 0, "whitespace_rescued": 0, "not_found": 0}
    for r in results:
        mt = r.get("match_type", "n/a")
        if mt in match_type_counts:
            match_type_counts[mt] += 1

    taxonomy = build_taxonomy(results)
    report_path = write_report(
        cfg=cfg,
        results=results,
        taxonomy=taxonomy,
        cross_chunk_count=total_cross,
        total_gold_spans=total_spans,
        match_type_counts=match_type_counts,
        report_name=report_name,
    )
    print(f"Taxonomy report written: {report_path}")
