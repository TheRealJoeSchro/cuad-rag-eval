"""
Smoke test for retrieval layer.

Runs in two modes:
  python -m retrieval.smoke_test --offline   # validates chunking + overlap logic only
  python -m retrieval.smoke_test             # full end-to-end with OpenAI (needs key)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import yaml
from retrieval.embedder import load_chunks, _prepend_overlap


def test_offline():
    print("=== Offline checks ===")
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    chunks = load_chunks(cfg["processed_dir"])
    print(f"Loaded {len(chunks)} chunks from {len(set(c['contract_id'] for c in chunks))} contracts")

    # Verify overlap prepend does not corrupt text
    sample = chunks[1]  # index 1 has a prev chunk
    prev = chunks[0]
    overlap = cfg.get("chunk_overlap", 64)
    embed_text = _prepend_overlap(sample["text"], prev["text"], overlap)

    assert sample["text"] in embed_text, "Canonical text must appear in embed text"
    assert embed_text != sample["text"], "Overlap should have added a prefix"
    print(f"Overlap prepend OK: '{embed_text[:60]}...'")

    # Spot-check span integrity for first 20 chunks
    contracts_path = Path(cfg["processed_dir"]) / "contracts.jsonl"
    contexts = {}
    with open(contracts_path) as f:
        for line in f:
            rec = json.loads(line)
            contexts[rec["contract_id"]] = rec["context"]

    mismatches = 0
    for c in chunks[:20]:
        ctx = contexts[c["contract_id"]]
        if ctx[c["char_start"]:c["char_end"]] != c["text"]:
            mismatches += 1
    print(f"Span integrity check (first 20 chunks): {mismatches} mismatches")

    # Collection name is deterministic
    from retrieval.store import _collection_name
    name1 = _collection_name(cfg)
    name2 = _collection_name(cfg)
    assert name1 == name2, "Collection name must be stable"
    print(f"Collection name: {name1}")

    print("Offline checks passed.\n")


def test_online(cfg):
    print("=== Online checks (OpenAI + Chroma) ===")
    from retrieval.retriever import Retriever

    retriever = Retriever(cfg)

    # Build index (skips already-indexed chunks)
    retriever.build_index()

    # Pick a contract and a known clause type
    contract_id = "ADAMSGOLFINC_03_21_2005-EX-10.17-ENDORSEMENT AGREEMENT"
    query = ('Highlight the parts (if any) of this contract related to "Governing Law" '
             'that should be reviewed by a lawyer. Details: Which state/country\'s law '
             'governs the contract?')

    results = retriever.search(contract_id=contract_id, query=query, k=5)
    print(f"Query: {query[:80]}...")
    print(f"Top-{len(results)} results for {contract_id}:")
    for i, r in enumerate(results):
        print(f"  [{i+1}] score={r['score']:.3f}  chars [{r['char_start']}:{r['char_end']}]")
        print(f"       {r['text'][:120].replace(chr(10), ' ')}...")
    print()

    # Load gold label for Governing Law in this contract
    contracts_path = Path(cfg["processed_dir"]) / "contracts.jsonl"
    with open(contracts_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["contract_id"] == contract_id:
                for label in rec["gold_labels"]:
                    if label["clause_type"] == "Governing Law" and label["answers"]:
                        gold = label["answers"][0]
                        print(f"Gold span: chars [{gold['answer_start']}:"
                              f"{gold['answer_start'] + len(gold['text'])}]")
                        print(f"Gold text: {repr(gold['text'][:120])}")
                        # Check if any result overlaps the gold span
                        for r in results:
                            if (r["char_start"] < gold["answer_start"] + len(gold["text"])
                                    and r["char_end"] > gold["answer_start"]):
                                print(f"  -> chunk_{r['chunk_index']} OVERLAPS gold span (recall hit)")
                        break

    print("Online checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    test_offline()

    if not args.offline:
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        test_online(cfg)
