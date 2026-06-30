"""
High-level retrieval interface.

Usage:
    retriever = Retriever(cfg)
    retriever.build_index()               # embed all chunks and upsert to Chroma
    results = retriever.search(           # top-k chunks for a contract + query
        contract_id="ADAMSGOLFINC_...",
        query="Highlight parts related to Governing Law...",
        k=5,
    )
"""

import os
from openai import OpenAI
from tqdm import tqdm

from retrieval.embedder import load_chunks, build_embed_inputs, embed_texts
from retrieval.store import (
    get_client,
    get_or_create_collection,
    upsert_chunks,
    query_collection,
    collection_chunk_ids,
)


class Retriever:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.embedding_model = cfg.get("embedding_model", "text-embedding-3-small")
        self.overlap_tokens = cfg.get("chunk_overlap", 64)
        self.k = cfg.get("retrieval_k", 5)
        self.processed_dir = cfg.get("processed_dir", "data/processed")
        self.chroma_dir = cfg.get("chroma_dir", ".chroma")

        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._client = get_client(self.chroma_dir)
        self._collection = get_or_create_collection(self._client, cfg)

    def build_index(self, force: bool = False) -> None:
        """
        Embed all chunks and upsert into the vector store.

        Skips chunks already indexed unless force=True.
        Overlap-prepend is applied here at embedding time.
        """
        chunks = load_chunks(self.processed_dir)

        if not force:
            already_indexed = collection_chunk_ids(self._collection)
            new_chunks = [c for c in chunks if c["chunk_id"] not in already_indexed]
        else:
            new_chunks = chunks

        if not new_chunks:
            print("All chunks already indexed. Pass force=True to re-embed.")
            return

        print(f"Embedding {len(new_chunks)} chunks (model: {self.embedding_model})...")

        # Build embed inputs with overlap prepend.
        # load_chunks already sorts by (contract_id, chunk_index).
        # For incremental indexing we need the previous chunk text per contract,
        # even if that chunk was already indexed. Rebuild full sorted list for context.
        all_sorted = load_chunks(self.processed_dir)
        prev_text_by_contract: dict[str, str] = {}
        new_chunk_ids = {c["chunk_id"] for c in new_chunks}

        embed_inputs: list[str] = []
        embed_chunks: list[dict] = []

        for chunk in all_sorted:
            cid = chunk["contract_id"]
            prev = prev_text_by_contract.get(cid)

            if chunk["chunk_id"] in new_chunk_ids:
                from retrieval.embedder import _prepend_overlap
                embed_text = _prepend_overlap(chunk["text"], prev, self.overlap_tokens)
                embed_inputs.append(embed_text)
                embed_chunks.append(chunk)

            prev_text_by_contract[cid] = chunk["text"]

        # Embed in batches
        vectors = embed_texts(embed_inputs, self.embedding_model, self._openai)
        print(f"Upserting {len(embed_chunks)} embeddings to Chroma...")
        upsert_chunks(self._collection, embed_chunks, vectors)
        print("Index build complete.")

    def search(
        self,
        contract_id: str,
        query: str,
        k: int | None = None,
    ) -> list[dict]:
        """
        Return top-k chunks from contract_id most relevant to query.

        Each result dict has: chunk_id, contract_id, text, char_start,
        char_end, chunk_index, total_chunks, token_count, score.
        """
        k = k or self.k
        # Embed the query (no overlap prepend for queries)
        resp = self._openai.embeddings.create(
            input=[query],
            model=self.embedding_model,
        )
        query_vector = resp.data[0].embedding

        return query_collection(
            self._collection,
            query_vector=query_vector,
            contract_id=contract_id,
            k=k,
        )