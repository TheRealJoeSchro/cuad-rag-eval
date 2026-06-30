"""
Embedding layer.

Handles overlap-prepend at embed time: each chunk's canonical text gets the tail
of the previous chunk prepended (token-limited to chunk_overlap) before embedding.
This gives the embedding model context continuity without corrupting char spans.

Only the embedding input changes; chunk metadata (char_start/char_end) is untouched.
"""

import os
import json
from pathlib import Path
import tiktoken
from openai import OpenAI


def _get_encoder():
    return tiktoken.get_encoding("cl100k_base")


def _prepend_overlap(text: str, prev_text: str | None, overlap_tokens: int) -> str:
    """Prepend tail of prev_text (up to overlap_tokens) to text for embedding."""
    if not prev_text or overlap_tokens <= 0:
        return text
    enc = _get_encoder()
    tokens = enc.encode(prev_text)
    tail_tokens = tokens[-overlap_tokens:] if len(tokens) > overlap_tokens else tokens
    tail = enc.decode(tail_tokens)
    return tail + "\n" + text


def embed_texts(texts: list[str], model: str, client: OpenAI) -> list[list[float]]:
    """Embed a batch of texts. Returns list of float vectors."""
    # OpenAI embedding API allows up to 2048 inputs per call; batch conservatively.
    BATCH = 512
    vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        resp = client.embeddings.create(input=batch, model=model)
        # Results are returned in order
        vectors.extend([item.embedding for item in resp.data])
    return vectors


def build_embed_inputs(
    chunks: list[dict],
    overlap_tokens: int,
) -> list[str]:
    """
    Produce the text to embed for each chunk.

    chunks must be sorted by (contract_id, chunk_index) so prev-chunk lookup is O(1).
    Returns one string per chunk — canonical text with optional overlap prefix.
    """
    inputs = []
    prev_text_by_contract: dict[str, str] = {}

    for chunk in chunks:
        cid = chunk["contract_id"]
        prev = prev_text_by_contract.get(cid)
        embed_text = _prepend_overlap(chunk["text"], prev, overlap_tokens)
        inputs.append(embed_text)
        prev_text_by_contract[cid] = chunk["text"]

    return inputs


def load_chunks(processed_dir: str) -> list[dict]:
    """Load all chunks from chunks.jsonl, sorted for stable overlap ordering."""
    path = Path(processed_dir) / "chunks.jsonl"
    chunks = []
    with open(path) as f:
        for line in f:
            chunks.append(json.loads(line))
    # Sort by contract then chunk index for deterministic overlap prepend
    chunks.sort(key=lambda c: (c["contract_id"], c["chunk_index"]))
    return chunks