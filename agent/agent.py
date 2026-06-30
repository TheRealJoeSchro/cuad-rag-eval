"""
Core clause-extraction agent.

For each (contract_id, clause_type, question):
  1. Retrieve top-k chunks from the vector store.
  2. Call LLM with grounding prompt.
  3. Parse JSON response; validate supporting_clause_text is in retrieved text.
  4. Return AgentResponse.
"""

import json
import re
import time

from agent.llm import call_llm
from agent.prompt import SYSTEM_PROMPT, build_user_prompt
from agent.schema import AgentResponse
from retrieval.retriever import Retriever


def _parse_json(text: str) -> dict | None:
    """Extract the first JSON object from LLM output."""
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        # Try to find a JSON object in the output
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def _validate_citation(supporting_text: str | None, chunks: list[dict]) -> str | None:
    """
    Return the chunk_id of the chunk that contains supporting_text verbatim.
    Returns None if no chunk contains it (hallucination guard).
    """
    if not supporting_text:
        return None
    for chunk in chunks:
        if supporting_text in chunk["text"]:
            return chunk["chunk_id"]
    return None


class Agent:
    def __init__(self, cfg: dict, retriever: Retriever):
        self.retriever = retriever
        self.provider = cfg.get("llm_provider", "anthropic")
        self.model = cfg.get("llm_model", "claude-haiku-4-5-20251001")
        self.max_tokens = cfg.get("max_tokens", 512)
        self.temperature = cfg.get("temperature", 0.0)
        self.k = cfg.get("retrieval_k", 5)

    def run(self, contract_id: str, clause_type: str, question: str) -> AgentResponse:
        t0 = time.monotonic()

        # Retrieve relevant chunks
        chunks = self.retriever.search(
            contract_id=contract_id,
            query=question,
            k=self.k,
        )
        retrieved_ids = [c["chunk_id"] for c in chunks]

        # Build and call LLM
        user_prompt = build_user_prompt(question, chunks)
        raw_text, in_tok, out_tok = call_llm(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        latency = time.monotonic() - t0

        # Parse response
        parsed = _parse_json(raw_text)

        if parsed is None:
            # Unparseable output — treat as absent
            return AgentResponse(
                contract_id=contract_id,
                clause_type=clause_type,
                question=question,
                answer=None,
                supporting_clause_text=None,
                source_citation=None,
                confidence=None,
                retrieved_chunk_ids=retrieved_ids,
                is_absent=True,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_s=round(latency, 3),
            )

        answer = parsed.get("answer")
        supporting = parsed.get("supporting_clause_text")
        citation = parsed.get("source_citation")
        confidence = parsed.get("confidence")

        # Treat explicit null/empty as absent
        is_absent = not answer or answer.strip().lower() in ("null", "none", "n/a", "")

        if is_absent:
            answer = None
            supporting = None
            citation = None
            confidence = None
        else:
            # Hallucination guard: verify supporting_clause_text is verbatim in a chunk.
            # If not, clear it so the eval correctly marks this as a citation failure
            # rather than a retrieval failure.
            verified_citation = _validate_citation(supporting, chunks)
            if verified_citation is None:
                # LLM fabricated text not in any retrieved chunk
                supporting = None
                citation = None
            else:
                # Use our verified citation (LLM may have gotten chunk_id wrong)
                citation = verified_citation

        return AgentResponse(
            contract_id=contract_id,
            clause_type=clause_type,
            question=question,
            answer=answer,
            supporting_clause_text=supporting,
            source_citation=citation,
            confidence=confidence,
            retrieved_chunk_ids=retrieved_ids,
            is_absent=is_absent,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=round(latency, 3),
        )