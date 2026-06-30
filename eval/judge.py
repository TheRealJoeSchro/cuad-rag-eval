"""
LLM-as-judge for secondary correctness scoring.

Uses a different model than the agent to avoid self-grading bias.
The rubric prompt is stored here and reproduced verbatim in the report.
"""

import os

JUDGE_RUBRIC = """\
You are evaluating a contract clause extraction system.

Gold standard clause text (expert-annotated):
{gold_text}

System's answer:
{agent_answer}

Question: Does the system's answer correctly identify the same clause \
and convey the same substantive meaning as the gold standard? \
Minor wording differences are acceptable. The answer must not \
contradict the gold standard or introduce material information \
not present in it.

Respond with exactly one line:
CORRECT: <one-line justification>
or
INCORRECT: <one-line justification>"""


def judge_answer(
    agent_answer: str,
    gold_answers: list[dict],
    provider: str,
    model: str,
) -> tuple[bool, str]:
    """
    Ask the judge LLM whether the agent's answer is correct.

    Returns (is_correct, justification).
    """
    from agent.llm import call_llm

    # Concatenate all gold span texts for multi-answer labels
    gold_text = "\n---\n".join(a["text"] for a in gold_answers)

    prompt = JUDGE_RUBRIC.format(gold_text=gold_text, agent_answer=agent_answer)

    text, _, _ = call_llm(
        system="You are a precise evaluator. Follow the instructions exactly.",
        user=prompt,
        provider=provider,
        model=model,
        max_tokens=128,
        temperature=0.0,
    )

    line = text.strip().split("\n")[0].strip()
    if line.upper().startswith("CORRECT"):
        justification = line.split(":", 1)[1].strip() if ":" in line else ""
        return True, justification
    else:
        justification = line.split(":", 1)[1].strip() if ":" in line else line
        return False, justification