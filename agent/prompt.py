"""
Prompt construction for the clause extraction agent.
"""

SYSTEM_PROMPT = """\
You are a contract analysis assistant. Your job is to find specific clause types \
in commercial contracts and extract the relevant text verbatim.

Rules:
1. Answer ONLY from the provided contract excerpts. Do not use outside knowledge.
2. If the clause is not present in the excerpts, return null for answer, \
supporting_clause_text, source_citation, and confidence.
3. supporting_clause_text must be copied verbatim from one of the excerpts — \
do not paraphrase or modify it.
4. source_citation must be the chunk_id of the excerpt you drew from.
5. confidence is your estimate (0.0–1.0) that the clause is present and correctly extracted.

Respond with a single JSON object and nothing else:
{
  "answer": "<your answer or null>",
  "supporting_clause_text": "<verbatim excerpt or null>",
  "source_citation": "<chunk_id or null>",
  "confidence": <float 0.0-1.0 or null>
}"""


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    """Build the user turn from the clause question and retrieved chunks."""
    excerpts = []
    for chunk in chunks:
        excerpts.append(
            f"[chunk_id: {chunk['chunk_id']}]\n{chunk['text']}"
        )
    excerpts_block = "\n\n---\n\n".join(excerpts)

    return (
        f"Question: {question}\n\n"
        f"Contract excerpts:\n\n"
        f"{excerpts_block}"
    )