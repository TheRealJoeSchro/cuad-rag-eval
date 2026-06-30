"""
Agent pipeline: run the clause-extraction agent over all (contract, clause_type)
pairs in the ingested dataset and write results.

Output file: agent_outputs_{run_name}.jsonl (or agent_outputs.jsonl if no run_name).
Each line is one AgentResponse serialised to JSON.
Existing outputs are skipped (resume-safe).
"""

import json
from pathlib import Path

from agent.agent import Agent
from retrieval.retriever import Retriever


def _output_filename(run_name: str | None) -> str:
    if run_name:
        return f"agent_outputs_{run_name}.jsonl"
    return "agent_outputs.jsonl"


def _load_contracts(processed_dir: str, contract_id_filter: str | None) -> list[dict]:
    path = Path(processed_dir) / "contracts.jsonl"
    contracts = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if contract_id_filter and rec["contract_id"] != contract_id_filter:
                continue
            contracts.append(rec)
    return contracts


def run_agent(cfg: dict, contract_id: str | None = None, run_name: str | None = None) -> None:
    processed_dir = cfg.get("processed_dir", "data/processed")
    out_path = Path(processed_dir) / _output_filename(run_name)

    # Load already-completed (contract_id, clause_type) pairs for resume
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                done.add((rec["contract_id"], rec["clause_type"]))

    contracts = _load_contracts(processed_dir, contract_id)
    if not contracts:
        print("No contracts found. Run `python cli.py ingest` first.")
        return

    # Build retriever (index must already exist from `cli.py run` → build_index)
    retriever = Retriever(cfg)
    agent = Agent(cfg, retriever)

    total_pairs = sum(len(c["gold_labels"]) for c in contracts)
    skip_count = sum(
        1 for c in contracts
        for label in c["gold_labels"]
        if (c["contract_id"], label["clause_type"]) in done
    )
    run_count = total_pairs - skip_count
    print(f"{total_pairs} (contract, clause) pairs | {skip_count} already done | {run_count} to run")

    with open(out_path, "a") as out_f:
        for contract in contracts:
            cid = contract["contract_id"]
            for label in contract["gold_labels"]:
                clause_type = label["clause_type"]
                if (cid, clause_type) in done:
                    continue

                resp = agent.run(
                    contract_id=cid,
                    clause_type=clause_type,
                    question=label["question"],
                )
                out_f.write(json.dumps(resp.to_dict()) + "\n")
                out_f.flush()

                status = "absent" if resp.is_absent else f"found (conf={resp.confidence or 0:.2f})"
                print(f"  {cid[:40]} / {clause_type:<35} -> {status}  "
                      f"[{resp.input_tokens}+{resp.output_tokens} tok, {resp.latency_s}s]")

    print(f"\nDone. Outputs written to {out_path}")