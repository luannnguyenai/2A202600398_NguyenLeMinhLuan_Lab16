"""
src/reflexion_lab/agents.py
-----------------------------
ReActAgent và ReflexionAgent theo chuẩn Shinn et al. 2023.

Hỗ trợ 2 mode:
- mode="real": dùng OllamaClient (llm_runtime.py).
- mode="mock": dùng mock_runtime.py (để test nhanh).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from rich.console import Console

from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

if TYPE_CHECKING:
    from .llm_runtime import OllamaClient

console = Console()


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

@dataclass
class BaseAgent:
    """
    Agent cơ sở chứa Reflexion loop hoàn chỉnh.

    Args:
        agent_type: "react" (1 attempt) hoặc "reflexion" (multi-attempt).
        max_attempts: Số lần thử tối đa.
        client: OllamaClient instance (chỉ dùng ở mode=real).
        mode: "real" dùng LLM thật, "mock" dùng mock_runtime.
    """

    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    client: Optional["OllamaClient"] = field(default=None, repr=False)
    mode: Literal["mock", "real"] = "real"

    def run(self, example: QAExample) -> RunRecord:
        """
        Chạy agent trên 1 example, trả về RunRecord đầy đủ.

        Flow (Reflexion paper, Shinn et al. 2023):
          for attempt in 1..max_attempts:
            Actor → answer
            Evaluator → judge
            if correct: break
            Reflector → reflection_entry
            append lesson vào reflection_memory
        """
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        total_tokens = 0
        total_latency = 0

        if self.mode == "real":
            from .llm_runtime import (
                OllamaClient,
                actor_answer,
                classify_failure_mode,
                evaluator,
                reflector,
            )
        else:
            from .mock_runtime import (  # type: ignore[assignment]
                actor_answer as _mock_actor,
                evaluator as _mock_evaluator,
                reflector as _mock_reflector,
                FAILURE_MODE_BY_QID,
            )

        prev_answer: Optional[str] = None
        looping = False

        for attempt_id in range(1, self.max_attempts + 1):
            # ── Adaptive max_attempts: skip nếu đã detect looping ──
            if looping:
                console.log(
                    f"[yellow]qid={example.qid} attempt={attempt_id}: "
                    "Detect looping — dừng sớm.[/yellow]"
                )
                break

            # ── Actor ──
            if self.mode == "real":
                assert self.client is not None, "OllamaClient chưa được truyền vào."
                ans, a_pt, a_ct, a_lat = actor_answer(
                    self.client, example, attempt_id, self.agent_type, reflection_memory
                )
                actor_tokens = a_pt + a_ct
                actor_latency = a_lat
            else:
                ans = _mock_actor(example, attempt_id, self.agent_type, reflection_memory)
                actor_tokens = 320 + attempt_id * 65 + (120 if self.agent_type == "reflexion" else 0)
                actor_latency = 160 + attempt_id * 40 + (90 if self.agent_type == "reflexion" else 0)

            # ── Detect looping (adaptive_max_attempts) ──
            if prev_answer is not None and ans.strip().lower() == prev_answer.strip().lower():
                looping = True

            prev_answer = ans

            # ── Evaluator ──
            if self.mode == "real":
                judge, e_pt, e_ct, e_lat = evaluator(self.client, example, ans)
                eval_tokens = e_pt + e_ct
                eval_latency = e_lat
            else:
                judge = _mock_evaluator(example, ans)
                eval_tokens = 200
                eval_latency = 80

            # Token & latency gộp actor + evaluator
            attempt_tokens = actor_tokens + eval_tokens
            attempt_latency = actor_latency + eval_latency

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=ans,
                score=judge.score,
                reason=judge.reason,
                token_estimate=attempt_tokens,
                latency_ms=attempt_latency,
            )
            total_tokens += attempt_tokens
            total_latency += attempt_latency
            final_answer = ans
            final_score = judge.score

            if judge.score == 1:
                # ── Adaptive max_attempts: thoát sớm khi đúng ──
                traces.append(trace)
                break

            # ── Reflexion loop ──
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                if self.mode == "real":
                    ref_entry, r_pt, r_ct, r_lat = reflector(
                        self.client, example, attempt_id, ans, judge
                    )
                    ref_tokens = r_pt + r_ct
                    ref_latency = r_lat
                else:
                    ref_entry = _mock_reflector(example, attempt_id, judge)
                    ref_tokens = 150
                    ref_latency = 60

                total_tokens += ref_tokens
                total_latency += ref_latency
                reflections.append(ref_entry)

                # Cập nhật reflection_memory (reflection_memory feature)
                memory_line = (
                    f"Attempt {attempt_id} failed: {ref_entry.failure_reason} "
                    f"| Lesson: {ref_entry.lesson} "
                    f"| Next: {ref_entry.next_strategy}"
                )
                reflection_memory.append(memory_line)

                # Gắn reflection vào trace
                trace = trace.model_copy(update={"reflection": ref_entry})

            traces.append(trace)

        # ── Failure mode ──
        if final_score == 1:
            failure_mode: str = "none"
        elif self.mode == "real":
            last_judge_reason = traces[-1].reason if traces else ""
            from .llm_runtime import classify_failure_mode

            failure_mode = classify_failure_mode(
                final_answer,
                example.gold_answer,
                traces[-1].reflection.failure_reason if traces[-1].reflection else last_judge_reason,  # type: ignore
            )
            # Override khi detect looping
            if looping and final_score == 0:
                failure_mode = "looping"
        else:
            failure_mode = FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,  # type: ignore[arg-type]
            reflections=reflections,
            traces=traces,
        )


# ---------------------------------------------------------------------------
# Concrete agents
# ---------------------------------------------------------------------------

class ReActAgent(BaseAgent):
    """ReAct Agent: 1 attempt duy nhất, không có Reflexion loop."""

    def __init__(
        self,
        client: Optional["OllamaClient"] = None,
        mode: Literal["mock", "real"] = "real",
    ) -> None:
        super().__init__(agent_type="react", max_attempts=1, client=client, mode=mode)


class ReflexionAgent(BaseAgent):
    """Reflexion Agent: tối đa 3 attempts với reflection loop đúng chuẩn paper."""

    def __init__(
        self,
        max_attempts: int = 3,
        client: Optional["OllamaClient"] = None,
        mode: Literal["mock", "real"] = "real",
    ) -> None:
        super().__init__(
            agent_type="reflexion",
            max_attempts=max_attempts,
            client=client,
            mode=mode,
        )
