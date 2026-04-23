"""
src/reflexion_lab/llm_runtime.py
----------------------------------
Runtime thật dùng Ollama local (llama3.1:8b).
Thay thế mock_runtime.py — giữ mock_runtime.py để test.

Tính năng:
- OllamaClient: wrap ollama.Client, đo token thật + latency thật.
- Retry 2 lần khi timeout / ValidationError.
- JSON parse robust bằng regex.
- Fallback token count bằng tiktoken.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import tiktoken
from pydantic import ValidationError
from rich.console import Console

import ollama

from .prompts import (
    ACTOR_SYSTEM,
    EVALUATOR_SYSTEM,
    REFLECTOR_SYSTEM,
    build_actor_user,
    build_evaluator_user,
    build_reflector_user,
)
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

console = Console()

# ---------------------------------------------------------------------------
# Token fallback encoder (gpt-4o-mini encoding ≈ llama3 token count)
# ---------------------------------------------------------------------------
_TIKTOKEN_ENC: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        try:
            _TIKTOKEN_ENC = tiktoken.encoding_for_model("gpt-4o-mini")
        except Exception:
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENC


def _count_tokens_fallback(text: str) -> int:
    """Đếm token bằng tiktoken khi Ollama API không trả về eval_count."""
    try:
        return len(_get_encoder().encode(text))
    except Exception:
        return len(text.split())


# ---------------------------------------------------------------------------
# JSON parse robust
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Trích xuất JSON object đầu tiên từ chuỗi bất kỳ.
    Hỗ trợ markdown fence và plain JSON.
    """
    # Thử parse thẳng
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Tìm block {...} đầu tiên
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Không tìm thấy JSON hợp lệ trong:\n{text[:500]}")


# ---------------------------------------------------------------------------
# Failure mode classifier
# ---------------------------------------------------------------------------

def classify_failure_mode(
    predicted: str,
    gold: str,
    judge: JudgeResult,
) -> str:
    """
    Phân loại lỗi thành 1 trong 5 failure mode dựa trên reason + answer text.
    """
    reason_lower = judge.reason.lower()
    pred_lower = normalize_answer(predicted)
    gold_lower = normalize_answer(gold)

    # Kiểm tra looping: answer giống nhau qua nhiều attempt (gọi từ agents.py)
    # Ở đây classify dựa trên reason
    if "loop" in reason_lower or "repeat" in reason_lower:
        return "looping"
    if "overfit" in reason_lower or "reflection" in reason_lower:
        return "reflection_overfit"
    if "hop" in reason_lower or "multi" in reason_lower or "second" in reason_lower:
        return "incomplete_multi_hop"
    if "entity" in reason_lower or "drift" in reason_lower or "wrong entity" in reason_lower:
        return "entity_drift"
    return "wrong_final_answer"


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Wrapper quanh ollama.Client để đo token thật và latency thật.

    Args:
        model: Tag Ollama model, mặc định 'llama3.1:8b'.
        host: Địa chỉ Ollama server.
        temperature: Nhiệt độ sinh văn bản (thấp = ổn định hơn).
        max_tokens: Số token tối đa trong response (num_predict).
        timeout: Timeout mỗi request (giây).
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: int = 180,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = ollama.Client(host=host)

    def chat(
        self,
        messages: list[dict],
        response_format: Optional[str] = None,
    ) -> tuple[str, int, int, int]:
        """
        Gửi chat request tới Ollama.

        Args:
            messages: List dicts {role, content}.
            response_format: Nếu "json" thì bật format=json.

        Returns:
            (text, prompt_tokens, completion_tokens, latency_ms)
        """
        options = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        fmt = "json" if response_format == "json" else None

        t0 = time.perf_counter()
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            format=fmt,  # type: ignore[arg-type]
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        text: str = resp.message.content or ""

        # Token thật từ Ollama response
        prompt_tokens: int = getattr(resp, "prompt_eval_count", 0) or 0
        completion_tokens: int = getattr(resp, "eval_count", 0) or 0

        # Fallback tiktoken nếu Ollama không trả về
        if prompt_tokens == 0:
            prompt_tokens = _count_tokens_fallback(
                " ".join(m.get("content", "") for m in messages)
            )
        if completion_tokens == 0:
            completion_tokens = _count_tokens_fallback(text)

        return text, prompt_tokens, completion_tokens, latency_ms


# ---------------------------------------------------------------------------
# Runtime functions (thay thế mock_runtime.py)
# ---------------------------------------------------------------------------

def _retry_chat(
    client: OllamaClient,
    messages: list[dict],
    response_format: Optional[str] = None,
    max_retries: int = 2,
) -> tuple[str, int, int, int]:
    """Gọi client.chat với retry 2 lần khi lỗi."""
    last_exc: Exception = RuntimeError("Unknown error")
    for attempt in range(max_retries + 1):
        try:
            return client.chat(messages, response_format=response_format)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                console.log(f"[yellow]Retry {attempt+1}/{max_retries} vì lỗi:[/yellow] {exc}")
                time.sleep(2 ** attempt)
    raise last_exc


def actor_answer(
    client: OllamaClient,
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int, int, int]:
    """
    Gọi Actor LLM để sinh câu trả lời.

    Returns:
        (answer, prompt_tokens, completion_tokens, latency_ms)
    """
    messages = [
        {"role": "system", "content": ACTOR_SYSTEM},
        {"role": "user", "content": build_actor_user(example, reflection_memory)},
    ]
    text, pt, ct, lat = _retry_chat(client, messages)

    # Trích xuất dòng "FINAL ANSWER: ..."
    match = re.search(r"FINAL ANSWER\s*:\s*(.+)", text, re.IGNORECASE)
    answer = match.group(1).strip() if match else text.strip().split("\n")[-1].strip()

    return answer, pt, ct, lat


def evaluator(
    client: OllamaClient,
    example: QAExample,
    answer: str,
) -> tuple[JudgeResult, int, int, int]:
    """
    Gọi Evaluator LLM để chấm điểm câu trả lời.

    Returns:
        (JudgeResult, prompt_tokens, completion_tokens, latency_ms)
    """
    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM},
        {"role": "user", "content": build_evaluator_user(example, answer)},
    ]

    for retry in range(3):
        try:
            text, pt, ct, lat = _retry_chat(client, messages, response_format="json")
            data = _extract_json(text)
            judge = JudgeResult.model_validate(data)
            return judge, pt, ct, lat
        except (ValidationError, ValueError, KeyError) as exc:
            console.log(f"[yellow]Evaluator parse lỗi (retry {retry}):[/yellow] {exc}")
            if retry == 2:
                # Fallback: so sánh normalize
                score = 1 if normalize_answer(example.gold_answer) == normalize_answer(answer) else 0
                return (
                    JudgeResult(
                        score=score,  # type: ignore[arg-type]
                        reason="Fallback: normalize_answer comparison (LLM parse failed).",
                        missing_evidence=[],
                        spurious_claims=[],
                    ),
                    0, 0, 0,
                )

    # Never reached, nhưng mypy cần
    raise RuntimeError("evaluator: unexpected exit")


def reflector(
    client: OllamaClient,
    example: QAExample,
    attempt_id: int,
    predicted: str,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, int, int, int]:
    """
    Gọi Reflector LLM để sinh reflection entry.

    Returns:
        (ReflectionEntry, prompt_tokens, completion_tokens, latency_ms)
    """
    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM},
        {"role": "user", "content": build_reflector_user(example, attempt_id, predicted, judge)},
    ]

    for retry in range(3):
        try:
            text, pt, ct, lat = _retry_chat(client, messages, response_format="json")
            data = _extract_json(text)
            entry = ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=data.get("failure_reason", judge.reason),
                lesson=data.get("lesson", ""),
                next_strategy=data.get("next_strategy", ""),
            )
            return entry, pt, ct, lat
        except (ValidationError, ValueError, KeyError) as exc:
            console.log(f"[yellow]Reflector parse lỗi (retry {retry}):[/yellow] {exc}")
            if retry == 2:
                return (
                    ReflectionEntry(
                        attempt_id=attempt_id,
                        failure_reason=judge.reason,
                        lesson="Fallback reflection: LLM parse failed.",
                        next_strategy="Re-read the context paragraphs carefully and complete all reasoning hops.",
                    ),
                    0, 0, 0,
                )

    raise RuntimeError("reflector: unexpected exit")
