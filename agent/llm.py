"""
Thin LLM client wrapper supporting Anthropic and OpenAI.

Returns (response_text, input_tokens, output_tokens).
"""

import os


def call_llm(
    system: str,
    user: str,
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, int]:
    """
    Call the configured LLM. Returns (text, input_tokens, output_tokens).
    """
    if provider == "anthropic":
        return _call_anthropic(system, user, model, max_tokens, temperature)
    elif provider == "openai":
        return _call_openai(system, user, model, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'anthropic' or 'openai'.")


def _call_anthropic(
    system: str,
    user: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, int]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = resp.content[0].text
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _call_openai(
    system: str,
    user: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, int]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content
    usage = resp.usage
    return text, usage.prompt_tokens, usage.completion_tokens