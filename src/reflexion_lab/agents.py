from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .mock_runtime import FAILURE_MODE_BY_QID
from .llm_runtime import actor_answer, evaluator, reflector, log_message
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        # Bonus: Adaptive Max Attempts based on difficulty
        current_max = self.max_attempts
        if example.difficulty == "hard":
            current_max = max(current_max, 5)
        elif example.difficulty == "medium":
            current_max = max(current_max, 3)
        elif example.difficulty == "easy":
            current_max = 1
            
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        log_message({"event": "process_start", "agent_type": self.agent_type, "qid": example.qid, "adaptive_max": current_max})
        for attempt_id in range(1, current_max + 1):
            log_message({"event": "attempt_start", "qid": example.qid, "attempt": attempt_id, "max": current_max})
            answer, tokens_actor, latency_actor = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            log_message({"event": "actor_done", "qid": example.qid, "answer": answer})
            
            judge, tokens_eval, latency_eval = evaluator(example, answer)
            log_message({"event": "evaluator_done", "qid": example.qid, "score": judge.score, "reason": judge.reason})
            
            # Use actual token count and latency from LLM response
            token_estimate = tokens_actor + tokens_eval
            latency_ms = latency_actor + latency_eval
            
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            
            if judge.score == 1:
                log_message({"event": "success", "qid": example.qid})
                traces.append(trace)
                break
            
            # Triển khai logic Reflexion tại đây
            if self.agent_type == 'reflexion' and attempt_id < self.max_attempts:
                log_message({"event": "reflector_start", "qid": example.qid})
                reflection, tokens_ref, latency_ref = reflector(example, attempt_id, answer, judge)
                log_message({"event": "reflector_done", "qid": example.qid, "next_strategy": reflection.next_strategy})
                reflection_memory.append(reflection.next_strategy)
                reflections.append(reflection)
                trace.token_estimate += tokens_ref
                trace.latency_ms += latency_ref
                trace.reflection = reflection
            
            traces.append(trace)
            
        log_message({"event": "process_finish", "qid": example.qid, "score": final_score})
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, difficulty=example.difficulty, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
