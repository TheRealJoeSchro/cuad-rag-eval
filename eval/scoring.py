"""
Scoring primitives for the eval harness.

All span operations work in the original contract text coordinate space.
"""

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Span location: find supporting_clause_text in raw contract text
# ---------------------------------------------------------------------------

def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """
    Collapse all whitespace runs to a single space.
    Returns (normalized_text, orig_positions) where orig_positions[i] is the
    index in the original text that normalized character i corresponds to.
    """
    orig_positions = []
    normalized = []
    i = 0
    while i < len(text):
        if text[i] in " \t\n\r":
            normalized.append(" ")
            orig_positions.append(i)
            while i < len(text) and text[i] in " \t\n\r":
                i += 1
        else:
            normalized.append(text[i])
            orig_positions.append(i)
            i += 1
    return "".join(normalized), orig_positions


def _find_all_occurrences(haystack: str, needle: str) -> list[tuple[int, int]]:
    """Return all (start, end) positions of needle in haystack."""
    results = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        results.append((idx, idx + len(needle)))
        start = idx + 1
    return results


def locate_citation(
    supporting_text: str,
    context: str,
    source_chunk_start: int | None,
    source_chunk_end: int | None,
) -> tuple[tuple[int, int] | None, str]:
    """
    Locate supporting_clause_text in raw contract text.

    Returns ((start, end), match_type) where match_type is one of:
      "exact"              — exact string match
      "whitespace_rescued" — matched after collapsing whitespace
      "not_found"          — no match at all

    Tie-break: pick the occurrence overlapping the agent's source chunk span.
    If that doesn't disambiguate, take first occurrence.
    Never uses gold spans.
    """
    if not supporting_text:
        return None, "not_found"

    # Phase 1: exact match
    occurrences = _find_all_occurrences(context, supporting_text)
    if occurrences:
        span = _pick_best_occurrence(occurrences, source_chunk_start, source_chunk_end)
        return span, "exact"

    # Phase 2: whitespace-normalized retry
    norm_ctx, ctx_map = _normalize_with_map(context)
    norm_query, _ = _normalize_with_map(supporting_text)

    norm_occurrences = _find_all_occurrences(norm_ctx, norm_query)
    if norm_occurrences:
        # Map back to original coordinates
        orig_occurrences = []
        for ns, ne in norm_occurrences:
            orig_start = ctx_map[ns]
            orig_end = ctx_map[ne - 1] + 1
            orig_occurrences.append((orig_start, orig_end))
        span = _pick_best_occurrence(orig_occurrences, source_chunk_start, source_chunk_end)
        return span, "whitespace_rescued"

    # Phase 3: not found
    return None, "not_found"


def _pick_best_occurrence(
    occurrences: list[tuple[int, int]],
    chunk_start: int | None,
    chunk_end: int | None,
) -> tuple[int, int]:
    """
    Pick the occurrence that overlaps the agent's source chunk span.
    If no chunk info or no overlap, take first occurrence.
    """
    if len(occurrences) == 1:
        return occurrences[0]

    if chunk_start is not None and chunk_end is not None:
        for s, e in occurrences:
            if s < chunk_end and e > chunk_start:
                return (s, e)

    return occurrences[0]


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def span_iou(pred: tuple[int, int], gold: tuple[int, int]) -> float:
    """Character-level IoU between two spans."""
    intersection = max(0, min(pred[1], gold[1]) - max(pred[0], gold[0]))
    union = (pred[1] - pred[0]) + (gold[1] - gold[0]) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def max_iou_vs_gold(pred_span: tuple[int, int], gold_answers: list[dict]) -> float:
    """Max IoU of pred_span against all gold answer spans."""
    if not gold_answers:
        return 0.0
    best = 0.0
    for ans in gold_answers:
        g_start = ans["answer_start"]
        g_end = g_start + len(ans["text"])
        best = max(best, span_iou(pred_span, (g_start, g_end)))
    return best


# ---------------------------------------------------------------------------
# Retrieval recall@k
# ---------------------------------------------------------------------------

def retrieval_recall(
    retrieved_chunk_ids: list[str],
    gold_answers: list[dict],
    chunks_by_id: dict[str, dict],
) -> bool:
    """True if any retrieved chunk overlaps any gold answer span."""
    if not gold_answers:
        return False
    for chunk_id in retrieved_chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        c_start, c_end = chunk["char_start"], chunk["char_end"]
        for ans in gold_answers:
            g_start = ans["answer_start"]
            g_end = g_start + len(ans["text"])
            if c_start < g_end and c_end > g_start:
                return True
    return False


# ---------------------------------------------------------------------------
# Cross-chunk gold span counting
# ---------------------------------------------------------------------------

def count_cross_chunk_spans(
    gold_answers: list[dict],
    contract_chunks: list[dict],
) -> tuple[int, int]:
    """
    Count how many gold spans straddle chunk boundaries.
    Returns (cross_chunk_count, total_span_count).
    """
    total = len(gold_answers)
    cross = 0
    for ans in gold_answers:
        g_start = ans["answer_start"]
        g_end = g_start + len(ans["text"])
        fully_contained = any(
            c["char_start"] <= g_start and c["char_end"] >= g_end
            for c in contract_chunks
        )
        if not fully_contained:
            cross += 1
    return cross, total


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_contracts(processed_dir: str) -> dict[str, dict]:
    """Load contracts keyed by contract_id."""
    path = Path(processed_dir) / "contracts.jsonl"
    contracts = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            contracts[rec["contract_id"]] = rec
    return contracts


def load_agent_outputs(processed_dir: str) -> list[dict]:
    path = Path(processed_dir) / "agent_outputs.jsonl"
    outputs = []
    with open(path) as f:
        for line in f:
            outputs.append(json.loads(line))
    return outputs


def load_chunks_by_id(processed_dir: str) -> dict[str, dict]:
    """Load chunks keyed by chunk_id."""
    path = Path(processed_dir) / "chunks.jsonl"
    chunks = {}
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c
    return chunks


def load_chunks_by_contract(processed_dir: str) -> dict[str, list[dict]]:
    """Load chunks grouped by contract_id."""
    path = Path(processed_dir) / "chunks.jsonl"
    by_contract: dict[str, list[dict]] = {}
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            by_contract.setdefault(c["contract_id"], []).append(c)
    return by_contract