# CUAD RAG Eval

Retrieval-grounded agent for clause-level question answering over commercial contracts,
evaluated against CUAD (Contract Understanding Atticus Dataset) expert annotations.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pull a small working subset (10 contracts) and ingest
make ingest N=10

# Run retrieval + agent over the subset
make run

# Score and produce report
make eval

# Full pipeline in one command
make all
```

Reports are written to `reports/`.

## Configuration

Edit `config.yaml` to swap LLM provider, embedding model, and retrieval parameters:

```yaml
embedding_model: text-embedding-3-small   # any OpenAI-compatible model
llm_model: claude-sonnet-4-6              # or gpt-4o, etc.
llm_provider: anthropic                   # anthropic | openai
retrieval_k: 5                            # top-k chunks per query
chunk_size: 512                           # tokens per chunk
chunk_overlap: 64
```

Set credentials as environment variables:
```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...       # only needed for OpenAI embeddings/LLM
```

## Dataset

CUAD: ~500 commercial contracts, 41 clause types, expert annotations.
Source: `https://huggingface.co/datasets/theatticusproject/cuad`

Raw contracts land in `data/raw/`, processed chunks in `data/processed/`.

## Metrics (headline)

| Metric | Method |
|---|---|
| Retrieval recall@k | Gold span overlaps any retrieved chunk |
| **Citation grounding (PRIMARY)** | IoU(supporting_clause_text, gold span) ≥ threshold, no LLM judge |
| Answer correctness (SECONDARY) | LLM-as-judge vs gold span, reported separately |
| False-positive rate | Agent claims clause present when CUAD label is empty |
| Latency / cost | Per-query wall time and token count |

All metrics split by clause-present vs clause-absent pairs.

## Project layout

```
ingest/       parse contracts, chunk clause-aware, write to data/processed/
retrieval/    embed chunks, build vector store, top-k search
agent/        grounded generation, structured JSON output
eval/         scoring harness, failure taxonomy, report generation
reports/      output tables and summaries (gitignored contents)
data/         raw contracts and processed chunks (gitignored)
config.yaml   all tunable parameters
cli.py        entry point for all commands
```