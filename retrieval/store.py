"""
ChromaDB vector store wrapper.

One persistent collection per embedding model + chunk config fingerprint,
so different configs don't collide. The collection stores chunk metadata
as Chroma document metadata, enabling contract-scoped searches.
"""

import hashlib
import json
import chromadb
from pathlib import Path


def _collection_name(cfg: dict) -> str:
    """
    Stable name derived from config knobs that affect the index.
    Changing embedding_model or chunk_size produces a different collection.
    """
    key = json.dumps({
        "embedding_model": cfg.get("embedding_model"),
        "chunk_size": cfg.get("chunk_size"),
        "chunk_overlap": cfg.get("chunk_overlap"),
    }, sort_keys=True)
    digest = hashlib.md5(key.encode()).hexdigest()[:8]
    model_slug = cfg.get("embedding_model", "unknown").replace("-", "_").replace(".", "_")
    return f"cuad_{model_slug}_{digest}"


def get_client(chroma_dir: str = ".chroma") -> chromadb.PersistentClient:
    Path(chroma_dir).mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=chroma_dir)


def get_or_create_collection(client: chromadb.PersistentClient, cfg: dict):
    name = _collection_name(cfg)
    # Chroma uses cosine similarity by default for normalized embedding vectors
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    collection,
    chunks: list[dict],
    vectors: list[list[float]],
) -> None:
    """Upsert chunk embeddings and metadata into the collection."""
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "contract_id": c["contract_id"],
            "char_start": c["char_start"],
            "char_end": c["char_end"],
            "chunk_index": c["chunk_index"],
            "total_chunks": c["total_chunks"],
            "token_count": c["token_count"],
        }
        for c in chunks
    ]
    documents = [c["text"] for c in chunks]

    # Chroma upsert in batches (API limit ~5000)
    BATCH = 1000
    for i in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[i:i + BATCH],
            embeddings=vectors[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
            documents=documents[i:i + BATCH],
        )


def query_collection(
    collection,
    query_vector: list[float],
    contract_id: str,
    k: int,
) -> list[dict]:
    """
    Return top-k chunks for a contract.

    Returns list of dicts with keys: chunk_id, contract_id, text,
    char_start, char_end, chunk_index, total_chunks, token_count, score.
    """
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        where={"contract_id": {"$eq": contract_id}},
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
        chunks.append({
            "chunk_id": chunk_id,
            "contract_id": meta["contract_id"],
            "text": doc,
            "char_start": meta["char_start"],
            "char_end": meta["char_end"],
            "chunk_index": meta["chunk_index"],
            "total_chunks": meta["total_chunks"],
            "token_count": meta["token_count"],
            "score": 1.0 - dist,  # cosine similarity (higher = better)
        })

    return chunks


def collection_chunk_ids(collection) -> set[str]:
    """Return the set of chunk_ids already indexed."""
    result = collection.get(include=[])
    return set(result["ids"])
