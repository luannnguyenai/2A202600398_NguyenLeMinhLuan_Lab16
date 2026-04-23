"""
src/reflexion_lab/prompts.py
------------------------------
System prompts (tiếng Anh) và helper builders cho Actor, Evaluator, Reflector.

Llama 3.1 8B hoạt động tốt hơn với tiếng Anh và JSON schema rõ ràng.
"""
from __future__ import annotations

from .schemas import JudgeResult, QAExample


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ACTOR_SYSTEM = """\
You are an expert multi-hop question-answering assistant.
You will be given a QUESTION and a set of CONTEXT paragraphs.
Your task:
1. Read all paragraphs carefully to gather evidence for EVERY reasoning hop.
2. Think step-by-step. Identify the first entity, then use it to find the second entity, and so on.
3. Cross-check your answer against the context before committing.
4. If prior REFLECTION NOTES are provided, use them to correct previous mistakes.
5. End your response with exactly one line in this format (do NOT omit):
   FINAL ANSWER: <your answer here>

Rules:
- The answer must be a short factual phrase or name — do NOT write a full sentence.
- Do NOT leave "FINAL ANSWER:" blank.
- Do NOT include markdown, bullet lists, or extra headers after the FINAL ANSWER line.
"""

EVALUATOR_SYSTEM = """\
You are a strict factual evaluator for multi-hop QA answers.

Given a QUESTION, GOLD ANSWER, and PREDICTED ANSWER, output ONLY valid JSON — NO markdown, NO commentary.

JSON schema (all fields required):
{
  "score": 0 or 1,
  "reason": "<brief explanation, max 2 sentences>",
  "missing_evidence": ["<evidence gap 1>", ...],
  "spurious_claims": ["<wrong claim 1>", ...]
}

Scoring rule:
- score=1 if and only if the predicted answer is semantically equivalent to the gold answer (allow minor spelling/case differences, articles, or abbreviations).
- score=0 otherwise.

Return ONLY the JSON object. No preamble, no markdown fences.
"""

REFLECTOR_SYSTEM = """\
You are an expert reasoning analyst for multi-hop QA agents.

Given a failed question-answer attempt, output ONLY valid JSON — NO markdown, NO commentary.

JSON schema (all fields required):
{
  "failure_reason": "<why the previous answer was wrong, max 2 sentences>",
  "lesson": "<what the agent missed or misunderstood, max 2 sentences>",
  "next_strategy": "<concrete action plan for the next attempt, be specific, max 3 sentences>"
}

The next_strategy must be actionable: name specific hops, entities, or paragraphs to focus on.
Return ONLY the JSON object.
"""


# ---------------------------------------------------------------------------
# User message builders
# ---------------------------------------------------------------------------

def build_actor_user(example: QAExample, reflection_memory: list[str]) -> str:
    """Xây dựng user message cho Actor, bao gồm context và reflection notes nếu có."""
    context_text = "\n\n".join(
        f"[{chunk.title}]\n{chunk.text}" for chunk in example.context
    )
    base = f"QUESTION: {example.question}\n\nCONTEXT:\n{context_text}"

    if reflection_memory:
        notes = "\n".join(f"- Attempt {i+1}: {note}" for i, note in enumerate(reflection_memory))
        base += f"\n\nREFLECTION NOTES FROM PREVIOUS ATTEMPTS:\n{notes}"

    base += "\n\nAnswer step-by-step, then end with:\nFINAL ANSWER: <your answer>"
    return base


def build_evaluator_user(example: QAExample, predicted: str) -> str:
    """Xây dựng user message cho Evaluator."""
    return (
        f"QUESTION: {example.question}\n"
        f"GOLD ANSWER: {example.gold_answer}\n"
        f"PREDICTED ANSWER: {predicted}\n\n"
        "Output JSON only."
    )


def build_reflector_user(
    example: QAExample,
    attempt_id: int,
    predicted: str,
    judge: JudgeResult,
) -> str:
    """Xây dựng user message cho Reflector."""
    context_text = "\n\n".join(
        f"[{chunk.title}]\n{chunk.text}" for chunk in example.context
    )
    missing = ", ".join(judge.missing_evidence) or "none identified"
    spurious = ", ".join(judge.spurious_claims) or "none identified"
    return (
        f"QUESTION: {example.question}\n"
        f"GOLD ANSWER: {example.gold_answer}\n"
        f"ATTEMPT {attempt_id} PREDICTED: {predicted}\n"
        f"EVALUATOR REASON: {judge.reason}\n"
        f"MISSING EVIDENCE: {missing}\n"
        f"SPURIOUS CLAIMS: {spurious}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        "Analyze the failure and output JSON only."
    )
