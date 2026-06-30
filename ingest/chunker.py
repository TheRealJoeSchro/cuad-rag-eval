"""
Clause-aware chunking for contract text.

Strategy:
1. Detect section boundaries via regex (numbered sections, ALL-CAPS headings).
2. Split at boundaries to produce offset-tracked section spans.
3. Merge small spans forward so short headings stay with their body.
4. Split oversized spans at paragraph breaks, then sentence breaks.

All operations work on (start, end) offset pairs into the original text.
Chunk text is always raw_text[start:end] — never modified, never stripped.
Overlap is not baked in; it's applied at embedding time by the retrieval layer.
"""

import re
import tiktoken

SECTION_BOUNDARY = re.compile(
    r'\n(?='
    r'(?:\d+\.(?:\d+\.?)*\s)'           # 1. / 1.1 / 10.3.1
    r'|(?:\([a-z]\)\s)'                  # (a) / (b)
    r'|(?:\([ivxlcdm]+\)\s)'            # (i) / (ii) / (iv)
    r'|(?:ARTICLE\s+[IVXLCDM]+)'        # ARTICLE I / ARTICLE IV
    r'|(?:Section\s+\d)'                 # Section 1 / Section 10
    r'|(?:SECTION\s+\d)'                # SECTION 1
    r'|(?:[A-Z][A-Z\s]{5,}(?:\n|$))'    # ALL-CAPS heading (6+ chars)
    r')'
)

_enc = None


def _get_encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def _token_count(text: str) -> int:
    return len(_get_encoder().encode(text))


Span = tuple[int, int]  # (start, end) offsets into original text


def _split_at_boundaries(text: str) -> list[Span]:
    """Split text at section boundaries. Returns list of (start, end) spans."""
    positions = [0]
    for m in SECTION_BOUNDARY.finditer(text):
        positions.append(m.start() + 1)  # +1: split is at the \n, new section starts after it
    positions.append(len(text))

    spans = []
    for i in range(len(positions) - 1):
        s, e = positions[i], positions[i + 1]
        if text[s:e].strip():
            spans.append((s, e))
    return spans


def _split_at_pattern(text: str, offset: int, pattern: re.Pattern) -> list[Span]:
    """Split a substring at a regex pattern, returning spans in original-text coordinates."""
    positions = [0]
    for m in pattern.finditer(text):
        positions.append(m.start())
    positions.append(len(text))

    spans = []
    for i in range(len(positions) - 1):
        s, e = positions[i], positions[i + 1]
        if text[s:e].strip():
            spans.append((offset + s, offset + e))
    return spans


_PARA_BREAK = re.compile(r'\n\s*\n')
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def _merge_small(text: str, spans: list[Span], min_tokens: int) -> list[Span]:
    """Merge consecutive spans when the current one is too small."""
    if not spans:
        return spans
    merged = [spans[0]]
    for s in spans[1:]:
        cur_start, cur_end = merged[-1]
        if _token_count(text[cur_start:cur_end]) < min_tokens:
            merged[-1] = (cur_start, s[1])
        else:
            merged.append(s)
    return merged


def _split_oversized(text: str, span: Span, max_tokens: int) -> list[Span]:
    """Break an oversized span into sub-spans respecting paragraph then sentence boundaries."""
    s, e = span
    chunk_text = text[s:e]

    if _token_count(chunk_text) <= max_tokens:
        return [span]

    # Try paragraph splits first
    para_spans = _split_at_pattern(chunk_text, s, _PARA_BREAK)
    if len(para_spans) > 1:
        result = []
        for ps in para_spans:
            result.extend(_split_oversized(text, ps, max_tokens))
        return _merge_small(text, result, max_tokens // 4)

    # Fall back to sentence splits
    sent_spans = _split_at_pattern(chunk_text, s, _SENTENCE_END)
    if len(sent_spans) <= 1:
        return [span]  # can't split further

    merged = [sent_spans[0]]
    for ss in sent_spans[1:]:
        cur_start, _ = merged[-1]
        candidate_end = ss[1]
        if _token_count(text[cur_start:candidate_end]) <= max_tokens:
            merged[-1] = (cur_start, candidate_end)
        else:
            merged.append(ss)
    return merged


def chunk_contract(
    text: str,
    contract_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """
    Chunk a contract into clause-aware pieces with character-span metadata.

    Returns list of dicts:
        chunk_id: str           "{contract_id}__chunk_{i}"
        contract_id: str
        text: str               exact slice raw_text[char_start:char_end]
        char_start: int         start offset in original text
        char_end: int           end offset in original text
        chunk_index: int
        total_chunks: int
        token_count: int
    """
    # Step 1-2: Split at section boundaries
    spans = _split_at_boundaries(text)

    # Step 3: Merge small sections
    min_tokens = chunk_size // 4
    spans = _merge_small(text, spans, min_tokens)

    # Step 4: Split oversized sections
    final_spans = []
    for span in spans:
        final_spans.extend(_split_oversized(text, span, chunk_size))

    # Build chunk records — text is always the exact raw slice
    total = len(final_spans)
    chunks = []
    for i, (s, e) in enumerate(final_spans):
        chunk_text = text[s:e]
        chunks.append({
            "chunk_id": f"{contract_id}__chunk_{i}",
            "contract_id": contract_id,
            "text": chunk_text,
            "char_start": s,
            "char_end": e,
            "chunk_index": i,
            "total_chunks": total,
            "token_count": _token_count(chunk_text),
        })

    return chunks