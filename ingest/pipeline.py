"""
Ingest pipeline: download CUAD data, chunk contracts, write to data/processed/.

Output:
  data/processed/chunks.jsonl    — one JSON object per chunk
  data/processed/contracts.jsonl — one JSON object per contract (id, title, gold labels)
"""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from ingest.chunker import chunk_contract

CUAD_REPO = "theatticusproject/cuad"
CUAD_FILE = "CUAD_v1/CUAD_v1.json"


def _load_cuad() -> dict:
    path = hf_hub_download(repo_id=CUAD_REPO, filename=CUAD_FILE, repo_type="dataset")
    with open(path) as f:
        return json.load(f)


def run_ingest(cfg: dict, subset: int | None = None) -> None:
    out_dir = Path(cfg.get("processed_dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = cfg.get("chunk_size", 512)
    chunk_overlap = cfg.get("chunk_overlap", 64)

    print("Downloading CUAD v1 annotations...")
    cuad = _load_cuad()
    articles = cuad["data"]

    if subset:
        articles = articles[:subset]

    print(f"Processing {len(articles)} contracts...")

    chunks_path = out_dir / "chunks.jsonl"
    contracts_path = out_dir / "contracts.jsonl"

    total_chunks = 0
    with open(chunks_path, "w") as cf, open(contracts_path, "w") as ctf:
        for art in articles:
            contract_id = art["title"]
            context = art["paragraphs"][0]["context"]
            qas = art["paragraphs"][0]["qas"]

            # Extract gold labels
            gold_labels = []
            for q in qas:
                clause_type = q["id"].split("__")[-1]
                gold_labels.append({
                    "clause_type": clause_type,
                    "question": q["question"],
                    "is_impossible": q["is_impossible"],
                    "answers": q["answers"],  # [{text, answer_start}, ...]
                })

            # Write contract record
            contract_record = {
                "contract_id": contract_id,
                "context_length": len(context),
                "gold_labels": gold_labels,
                "context": context,
            }
            ctf.write(json.dumps(contract_record) + "\n")

            # Chunk and write
            chunks = chunk_contract(
                text=context,
                contract_id=contract_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for chunk in chunks:
                cf.write(json.dumps(chunk) + "\n")
            total_chunks += len(chunks)

            print(f"  {contract_id}: {len(context)} chars -> {len(chunks)} chunks")

    print(f"\nDone. {len(articles)} contracts, {total_chunks} chunks.")
    print(f"  {chunks_path}")
    print(f"  {contracts_path}")