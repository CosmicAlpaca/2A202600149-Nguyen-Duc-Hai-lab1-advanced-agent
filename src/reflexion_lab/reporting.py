from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from .schemas import ReportPayload, RunRecord

def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {"count": len(rows), "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4), "avg_attempts": round(mean(r.attempts for r in rows), 4), "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2), "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2)}
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {"em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4), "attempts_abs": round(summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"], 4), "tokens_abs": round(summary["reflexion"]["avg_token_estimate"] - summary["react"]["avg_token_estimate"], 2), "latency_abs": round(summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"], 2)}
    return summary

def summarize_by_difficulty(records: list[RunRecord]) -> dict:
    grouped: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.agent_type, record.difficulty)].append(record)
    
    diff_summary = {}
    for (agent, diff), rows in grouped.items():
        if agent not in diff_summary: diff_summary[agent] = {}
        diff_summary[agent][diff] = {
            "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4),
            "count": len(rows)
        }
    return diff_summary

def failure_breakdown(records: list[RunRecord]) -> dict:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        grouped[record.agent_type][record.failure_mode] += 1
    return {agent: dict(counter) for agent, counter in grouped.items()}

def build_report(records: list[RunRecord], dataset_name: str, mode: str = "mock") -> ReportPayload:
    examples = [{"qid": r.qid, "agent_type": r.agent_type, "difficulty": r.difficulty, "gold_answer": r.gold_answer, "predicted_answer": r.predicted_answer, "is_correct": r.is_correct, "attempts": r.attempts, "failure_mode": r.failure_mode, "reflection_count": len(r.reflections)} for r in records]
    return ReportPayload(
        meta={"dataset": dataset_name, "mode": mode, "num_records": len(records), "agents": sorted({r.agent_type for r in records})},
        summary=summarize(records),
        difficulty_summary=summarize_by_difficulty(records),
        failure_modes=failure_breakdown(records),
        examples=examples,
        extensions=["adaptive_max_attempts", "difficulty_breakdown", "structured_evaluator", "reflection_memory", "benchmark_report_json"],
        discussion="Reflexion helps most on 'hard' and 'medium' questions where multi-hop reasoning is required. Easy questions often pass on the first attempt."
    )

def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    s = report.summary
    ds = report.difficulty_summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    
    # Build difficulty table
    diff_md = "| Difficulty | Agent | Count | EM Score |\n|---|---|---|---:|\n"
    for agent in sorted(ds.keys()):
        for diff in ["easy", "medium", "hard"]:
            if diff in ds[agent]:
                stats = ds[agent][diff]
                diff_md += f"| {diff.capitalize()} | {agent.upper()} | {stats['count']} | {stats['em']} |\n"

    ext_lines = "\n".join(f"- {item}" for item in report.extensions)
    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg token estimate | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## Difficulty Breakdown
{diff_md}

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
