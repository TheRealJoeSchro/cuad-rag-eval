"""
Output schema for agent responses.

Every (contract, clause_type) query produces one AgentResponse.
null fields signal "clause not found" — the agent must not hallucinate.
"""

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class AgentResponse:
    contract_id: str
    clause_type: str
    question: str

    # Core output — null when agent determines clause is absent
    answer: str | None                  # free-text answer
    supporting_clause_text: str | None  # verbatim excerpt from retrieved chunk
    source_citation: str | None         # chunk_id of the source chunk
    confidence: float | None            # 0.0–1.0, null when absent

    # Provenance
    retrieved_chunk_ids: list[str]      # all chunk_ids fed to the LLM
    is_absent: bool                     # True when agent returns null answer

    # Cost tracking
    input_tokens: int
    output_tokens: int
    latency_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
