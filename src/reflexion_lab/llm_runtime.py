import json
import time
import os
import urllib.request
import urllib.error
from typing import Tuple
from dotenv import load_dotenv
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .utils import normalize_answer

load_dotenv()

GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEY", "").split(",") if k.strip()]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_key_index = 0

def log_message(message: dict | str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(message, str):
        message = {"info": message}
    log_entry = {"timestamp": timestamp, **message}
    with open("agent.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    print(json.dumps(log_entry, ensure_ascii=False))

def call_groq(prompt: str, system_instruction: str = None, expect_json: bool = False, retries: int = 5) -> Tuple[str, int, int]:
    global _key_index
    start_time = time.time()
    
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": GROQ_MODEL,
        "messages": messages
    }
    
    if expect_json:
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload).encode("utf-8")
    
    last_error = ""
    for i in range(retries):
        if not GROQ_API_KEYS:
            return "No API keys found in .env", 0, 0
            
        current_key = GROQ_API_KEYS[_key_index % len(GROQ_API_KEYS)]
        req = urllib.request.Request(GROQ_URL, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {current_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                latency_ms = int((time.time() - start_time) * 1000)
                
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = result.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                
                log_message({"event": "api_success", "tokens": total_tokens, "latency_ms": latency_ms})
                return text, total_tokens, latency_ms
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rotate key on 429 too
                _key_index += 1
                wait_time = (2 ** i) + 1
                log_message({"event": "api_retry", "error": "429", "wait_s": wait_time, "attempt": i+1, "next_key_index": _key_index % len(GROQ_API_KEYS)})
                time.sleep(wait_time)
                continue
            if e.code == 403 or e.code == 401:
                # Invalid key? Try rotating
                _key_index += 1
                log_message({"event": "api_retry", "error": str(e.code), "info": "Key forbidden/unauthorized, rotating...", "next_key_index": _key_index % len(GROQ_API_KEYS)})
                if len(GROQ_API_KEYS) > 1:
                    continue
            last_error = f"HTTP Error {e.code}: {e.reason}"
            break
        except Exception as e:
            last_error = str(e)
            break
            
    latency_ms = int((time.time() - start_time) * 1000)
    log_message({"event": "api_fail", "error": last_error})
    return last_error, 0, latency_ms
            
    latency_ms = int((time.time() - start_time) * 1000)
    log_message({"event": "api_fail", "error": last_error})
    return last_error, 0, latency_ms

def format_context(context_chunks):
    return "\n".join([f"[{chunk.title}] {chunk.text}" for chunk in context_chunks])

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> Tuple[str, int, int]:
    context_str = format_context(example.context)
    prompt = f"Context:\n{context_str}\n\nQuestion: {example.question}\n"
    
    if reflection_memory:
        prompt += "\nPrevious Strategies & Reflections:\n" + "\n".join(reflection_memory)
    
    prompt += "\nPlease provide a short, clear final answer based on the context."
    
    answer_text, tokens, latency = call_groq(prompt, system_instruction=ACTOR_SYSTEM, expect_json=False)
    return answer_text.strip(), tokens, latency

def evaluator(example: QAExample, answer: str) -> Tuple[JudgeResult, int, int]:
    # Saving tokens: Deterministic matching instead of LLM call
    gold = normalize_answer(example.gold_answer)
    pred = normalize_answer(answer)
    
    is_correct = gold in pred or pred in gold
    reason = "Exact or normalized match found." if is_correct else "Answer does not match gold answer."
    
    return JudgeResult(score=1 if is_correct else 0, reason=reason, missing_evidence=[], spurious_claims=[]), 0, 0

def reflector(example: QAExample, attempt_id: int, answer: str, judge: JudgeResult) -> Tuple[ReflectionEntry, int, int]:
    context_str = format_context(example.context)
    prompt = (
        f"Context:\n{context_str}\n\n"
        f"Question: {example.question}\n"
        f"Previous Answer: {answer}\n"
        f"Evaluator Feedback: {judge.reason}\n\n"
        "Please return a JSON object with failure_reason, lesson, and next_strategy."
    )
    
    text, tokens, latency = call_groq(prompt, system_instruction=REFLECTOR_SYSTEM, expect_json=True)
    try:
        data = json.loads(text)
        return ReflectionEntry(attempt_id=attempt_id, **data), tokens, latency
    except Exception:
        return ReflectionEntry(attempt_id=attempt_id, failure_reason="Parse failed", lesson="None", next_strategy="Try again"), tokens, latency
