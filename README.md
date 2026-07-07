# CUAD RAG Eval

## Overview
The system takes in contracts from the CUAD dataset and chunks each contract into passages. When a question is asked about a clause type (like governing law or termination, etc), the agent searches that contract's own chunks to locate the relevant passage and answers accordingly, while citing the supporting text. I evaluated the accuracy and drawbacks of different models and retrieval approaches against CUAD's expert annotations. To improve the agent, I tested better models and also tested different retrieval techniques. When I tried using better models, the hallucinations went down noticeably. The different retrieval techniques lifted the model recall, but the headline score didn't move because the extra clauses it recovered just failed at the next stage, which was the extraction stage, instead. Retrieval got better, but the final number stayed flat. The thing that neither a better model nor more retrieval fixed is the agent's verbatim citations - Even when the right clause was successfully retrieved and sitting in front of the agent, it ended up quoting the wrong text as its supporting evidence. This points to the extraction quality being the biggest constraint of the system. It caps the score regardless of other improvements.

## How it works
There are four stages in the pipeline, starting with ingestion. The contracts are parsed and chunked into passages, splitting on structural boundaries, i.e., numbered sections and headings, rather than fixed-size cuts. This allows a clause to stay intact in one chunk, rather than getting sliced across two. Each chunk stores its exact character offsets back into the original contract, which makes citation scoring possible later on. Retrieval embeds every chunk with OpenAI’s text-embedding-3-small into a local ChromaDB store. At query time it embeds the clause question and pulls the top-k most similar chunks. The agent receives those chunks with a grounding prompt that requires it to answer only from the provided text, quote the supporting clause verbatim, and return null if the clause isn’t present, all as structured JSON. The eval harness scores each answer against CUAD’s expert spans. It locates the agent’s quoted text back in the raw contract, computes overlap based on characters against the gold span, and counts a hit above 0.5. It splits results by whether the clause is actually present, tracks false positives on absent clauses, runs a secondary LLM-judge, and buckets every failure by cause. 

## Results

10 contracts, 410 query pairs (132 present clauses, 278 absent).

| Metric | Baseline (Haiku, k=5) | Sonnet, k=5 | Haiku, k=10 |
|---|---|---|---|
| Citation grounding (primary) | 27.3% | 24.2% | 27.3% |
| Retrieval recall@k | 65.2% | 65.2% | 76.5% |
| False-positive rate | 32.4% | 25.5% | 33.8% |
| LLM-judge correctness | 24.6% | 17.1% | 31.9% |
| Cost vs baseline | 1.0x | 2.9x | 1.5x |

Higher is better for everything except false-positive rate and cost.

Two things stand out. Moving from Haiku to Sonnet cut the false-positive rate by
6.8 points (32.4% to 25.5%) but didn't improve citation grounding. Bumping k from
5 to 10 lifted retrieval recall by 11.4 points (65.2% to 76.5%) but left citation
grounding flat, since the newly retrieved clauses failed at extraction instead.

## Worked examples

Three real cases from the baseline run, one per outcome.

**Success (IoU 1.0).** Governing Law clause, Loha Company contract.
Question: what law governs the contract?
Agent answer: governed by the law of the People's Republic of China, otherwise
by the UN Convention on Contracts for the International Sale of Goods.
Agent cited: "It will be governed by the law of the People's Republic of China,
otherwise it is governed by United Nations Convention on Contract for the
International Sale of Goods."
Gold span: identical text.
The retrieved chunk contained the clause, and the agent quoted it verbatim. Cited
text matches the expert span exactly, so IoU is 1.0.

**Wrong extraction (recall hit, IoU 0.0).** Effective Date clause, Lime Energy
distributor agreement.
The right chunk was retrieved (recall succeeded), but the agent got lost. It
pointed at "Section 4.1" and a "Section 1.3" it claimed wasn't in the excerpts,
then declined to give a citation at all. The gold span was the term/commencement
clause sitting in the retrieved text. Retrieval did its job; the agent failed to
extract the answer that was in front of it. This is the biggest failure bucket.

**Hallucination (absent clause, false positive).** Notice Period To Terminate
Renewal, Lime Energy distributor agreement.
This clause does not exist in the contract. Instead of saying so, the agent
asserted "30 days' notice to terminate, as stated in Section 4.2," inventing a
specific figure and citing a section that does not support it. The correct answer
was to abstain. This is the failure a deployment team fears most: a confident,
specific answer to a question whose real answer is "not present."

## Scope and next steps

These results are a v1 on a 10-contract subset (410 query pairs). The goal was to
prove the eval harness works and surface where the system actually breaks, not to
post a full benchmark. The numbers are directional, not final.

The next steps:

- Scale to the full CUAD corpus (~500 contracts) to confirm the bottlenecks hold.
- Attack the retrieval ceiling with a legal-domain-tuned embedding model rather than
  just raising k, since raising k lifted recall but not the headline score.
- Test whether a stronger agent model breaks the extraction wall that neither Sonnet
  nor k=10 could move. Extraction, not retrieval or hallucination, is the binding
  constraint, so that is where the next real gain has to come from.

The harness is already built to measure all three.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set credentials
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

# Pull a small subset (10 contracts), ingest, run, and score
make all N=10
```

Reports are written to `reports/`. `make all` runs ingest, agent, and eval end to end; the individual steps are `make ingest N=10`, `make run`, `make eval`.

## Configuration

Edit `config.yaml` to swap models and retrieval parameters:

```yaml
embedding_model: text-embedding-3-small
llm_model: claude-haiku-4-5
llm_provider: anthropic
retrieval_k: 5
chunk_size: 512
chunk_overlap: 64
iou_threshold: 0.5
```

## Dataset

CUAD (Contract Understanding Atticus Dataset): ~500 commercial contracts, 41 clause
types, expert lawyer annotations as ground truth.
Source: https://huggingface.co/datasets/theatticusproject/cuad

## Project layout

```
ingest/       parse contracts, chunk clause-aware, write to data/processed/
retrieval/    embed chunks, build vector store, top-k search
agent/        grounded generation, structured JSON output
eval/         scoring harness, failure taxonomy, report generation
reports/      output tables and comparison reports
data/         processed chunks (raw contracts gitignored)
config.yaml   all tunable parameters
cli.py        entry point for all commands
```