"""
Phase 3 - Routing Recommendation Generator

Purpose:
- Read Phase 1 performance summary and Phase 2 quality summary
- Merge model/task candidates
- Generate routing recommendations for llm-router
- Save results under ./phase3_results

Default inputs:
- ./phase1_results/phase1_summary.csv
- ./phase2_results/phase2_summary.csv

Fallback inputs:
- ./phase1_summary.csv
- ./phase2_summary.csv

Default outputs:
- ./phase3_results/routing_recommendations.json
- ./phase3_results/routing_candidates.csv
- ./phase3_results/routing_recommendations.md

Run:
    python phase3-routing-recommender.py

Optional env overrides:
    set PHASE1_SUMMARY_CSV=./phase1_results/phase1_summary.csv
    set PHASE2_SUMMARY_CSV=./phase2_results/phase2_summary.csv
    set PHASE3_RESULTS_DIR=./phase3_results
    set PHASE3_QUALITY_TOLERANCE=0.25
    set PHASE3_PASS_THRESHOLD=7.0
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# 1. Configuration
# ============================================================

PHASE1_SUMMARY_CSV = Path(os.getenv("PHASE1_SUMMARY_CSV", "./phase1_results/phase1_summary.csv"))
PHASE2_SUMMARY_CSV = Path(os.getenv("PHASE2_SUMMARY_CSV", "./phase2_results/phase2_summary.csv"))
OUTPUT_DIR = Path(os.getenv("PHASE3_RESULTS_DIR", "./phase3_results"))

QUALITY_TOLERANCE = float(os.getenv("PHASE3_QUALITY_TOLERANCE", "0.25"))
PASS_THRESHOLD = float(os.getenv("PHASE3_PASS_THRESHOLD", "7.0"))

ROUTING_RECOMMENDATIONS_JSON = OUTPUT_DIR / "routing_recommendations.json"
ROUTING_CANDIDATES_CSV = OUTPUT_DIR / "routing_candidates.csv"
ROUTING_RECOMMENDATIONS_MD = OUTPUT_DIR / "routing_recommendations.md"

# Task policy:
# - prefer_efficiency: within quality tolerance, choose faster/lighter candidate.
# - prefer_quality: choose highest quality as primary; optionally choose fastest passing model as fast.
# - balanced: choose efficient candidate if close enough to highest quality.
# - preferred_model: hard preference when model exists and passes, useful for coding/debug implementation role.
TASK_POLICIES: Dict[str, Dict[str, Any]] = {
    "router": {
        "mode": "prefer_efficiency",
        "quality_tolerance": 0.30,
        "roles": ["primary", "fallback"],
    },
    "short_question": {
        "mode": "prefer_efficiency",
        "quality_tolerance": 0.30,
        "roles": ["primary", "fallback"],
    },
    "summarization": {
        "mode": "balanced",
        "quality_tolerance": 0.25,
        "roles": ["primary", "fallback"],
    },
    "draft_generation": {
        "mode": "prefer_quality",
        "quality_tolerance": 0.20,
        "roles": ["primary", "fast"],
    },
    "analysis": {
        "mode": "balanced",
        "quality_tolerance": 0.25,
        "roles": ["primary", "quality"],
    },
    "knowledge_refine": {
        "mode": "balanced",
        "quality_tolerance": 0.25,
        "roles": ["primary", "quality"],
    },
    "prompt_engineering": {
        "mode": "balanced",
        "quality_tolerance": 0.25,
        "roles": ["primary", "quality"],
    },
    "coding": {
        "mode": "prefer_quality",
        "preferred_model": "deepseek-coder:6.7b-instruct",
        "roles": ["primary"],
    },
    "debug": {
        "mode": "prefer_quality",
        "preferred_model": "deepseek-coder:6.7b-instruct",
        "roles": ["primary", "explanation"],
        "explanation_model_preference": "mistral:7b-instruct",
    },
}


# ============================================================
# 2. Utilities
# ============================================================

def pick_existing_path(preferred: Path, fallback: Path) -> Path:
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"找不到檔案：{preferred} 或 {fallback}")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in {"true", "false"}:
                return 1.0 if value.lower() == "true" else 0.0
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass"}


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def norm_ratio(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))


def inverse_norm_ratio(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (value / max_value)))


def file_stem_name(path_like: str) -> str:
    if not path_like:
        return ""
    return Path(path_like).name


def first_non_empty(row: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] not in {"", None}:
            return row[key]
    return default


# ============================================================
# 3. Data Model
# ============================================================

@dataclass
class Candidate:
    task: str
    model: str
    quality: float
    raw_quality: float
    tps: float
    vram_max_mb: float
    gpu_util_avg: float
    temp_peak_c: float
    load_sec: float
    wall_sec: float
    passed: bool
    confidence: float
    source_file: str
    primary_judge: str
    secondary_judge: str
    reason: str
    suggested_improvement: str
    efficiency_score: float = 0.0
    composite_score: float = 0.0

    def to_row(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "model": self.model,
            "quality": round(self.quality, 2),
            "raw_quality": round(self.raw_quality, 2),
            "tps": round(self.tps, 2),
            "vram_max_mb": round(self.vram_max_mb, 2),
            "gpu_util_avg": round(self.gpu_util_avg, 2),
            "temp_peak_c": round(self.temp_peak_c, 2),
            "load_sec": round(self.load_sec, 2),
            "wall_sec": round(self.wall_sec, 2),
            "pass": self.passed,
            "confidence": round(self.confidence, 2),
            "efficiency_score": round(self.efficiency_score, 4),
            "composite_score": round(self.composite_score, 4),
            "source_file": self.source_file,
            "primary_judge": self.primary_judge,
            "secondary_judge": self.secondary_judge,
            "reason": self.reason,
            "suggested_improvement": self.suggested_improvement,
        }


# ============================================================
# 4. Merge Phase 1 + Phase 2
# ============================================================

def build_phase1_lookup(phase1_rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in phase1_rows:
        model = str(first_non_empty(row, "Model", "model")).strip()
        task = str(first_non_empty(row, "Task", "task")).strip()
        if model and task:
            lookup[(model, task)] = row

        output_json = file_stem_name(str(first_non_empty(row, "Output_JSON", "output_json")))
        if output_json:
            lookup[(output_json, task)] = row
    return lookup


def make_candidates(phase1_rows: List[Dict[str, str]], phase2_rows: List[Dict[str, str]]) -> List[Candidate]:
    phase1_lookup = build_phase1_lookup(phase1_rows)

    candidates: List[Candidate] = []
    for row in phase2_rows:
        model = str(first_non_empty(row, "model", "Model")).strip()
        task = str(first_non_empty(row, "task", "Task")).strip()
        source_file = file_stem_name(str(first_non_empty(row, "source_file", "Source_File")))

        p1 = phase1_lookup.get((model, task)) or phase1_lookup.get((source_file, task)) or {}

        quality = as_float(first_non_empty(row, "weighted_final_score", "quality_score", default=0))
        raw_quality = as_float(first_non_empty(row, "raw_quality_score", "raw_overall_quality", default=quality))
        tps = as_float(first_non_empty(row, "tps", "TPS", default=first_non_empty(p1, "TPS", default=0)))
        vram = as_float(first_non_empty(row, "vram_max_mb", "VRAM_Max_MB", default=first_non_empty(p1, "VRAM_Max_MB", default=0)))
        gpu_util = as_float(first_non_empty(row, "gpu_util_avg", "GPU_Util_Avg", default=first_non_empty(p1, "GPU_Util_Avg", default=0)))
        temp = as_float(first_non_empty(row, "temp_peak_c", "Temp_Peak_C", default=first_non_empty(p1, "Temp_Peak_C", default=0)))
        load = as_float(first_non_empty(row, "load_sec", "Load_Sec", default=first_non_empty(p1, "Load_Sec", default=0)))
        wall = as_float(first_non_empty(row, "wall_sec", "Wall_Sec", default=first_non_empty(p1, "Wall_Sec", default=0)))

        candidates.append(
            Candidate(
                task=task,
                model=model,
                quality=quality,
                raw_quality=raw_quality,
                tps=tps,
                vram_max_mb=vram,
                gpu_util_avg=gpu_util,
                temp_peak_c=temp,
                load_sec=load,
                wall_sec=wall,
                passed=as_bool(first_non_empty(row, "pass", "Pass", default=quality >= PASS_THRESHOLD)),
                confidence=as_float(first_non_empty(row, "confidence", "Confidence", default=0)),
                source_file=source_file,
                primary_judge=str(first_non_empty(row, "primary_judge", default="")),
                secondary_judge=str(first_non_empty(row, "secondary_judge", default="")),
                reason=str(first_non_empty(row, "reason", default="")),
                suggested_improvement=str(first_non_empty(row, "suggested_improvement", default="")),
            )
        )

    return score_candidates(candidates)


def score_candidates(candidates: List[Candidate]) -> List[Candidate]:
    if not candidates:
        return []

    max_tps = max((c.tps for c in candidates), default=0.0)
    max_vram = max((c.vram_max_mb for c in candidates), default=0.0)
    max_wall = max((c.wall_sec for c in candidates), default=0.0)

    for c in candidates:
        tps_norm = norm_ratio(c.tps, max_tps)
        vram_inverse = inverse_norm_ratio(c.vram_max_mb, max_vram)
        wall_inverse = inverse_norm_ratio(c.wall_sec, max_wall)

        # Efficiency favors speed, then VRAM, then shorter wall time.
        c.efficiency_score = round((tps_norm * 0.60) + (vram_inverse * 0.25) + (wall_inverse * 0.15), 4)

        # Composite score is only a helper. Per-task policy still decides roles.
        quality_norm = max(0.0, min(1.0, c.quality / 10.0))
        c.composite_score = round((quality_norm * 0.70) + (c.efficiency_score * 0.30), 4)

    return candidates


# ============================================================
# 5. Recommendation Logic
# ============================================================

def candidates_by_task(candidates: List[Candidate]) -> Dict[str, List[Candidate]]:
    grouped: Dict[str, List[Candidate]] = {}
    for c in candidates:
        grouped.setdefault(c.task, []).append(c)
    return grouped


def sort_by_quality(candidates: Iterable[Candidate]) -> List[Candidate]:
    return sorted(
        candidates,
        key=lambda c: (c.quality, c.confidence, c.tps, -c.vram_max_mb),
        reverse=True,
    )


def sort_by_efficiency(candidates: Iterable[Candidate]) -> List[Candidate]:
    return sorted(
        candidates,
        key=lambda c: (c.efficiency_score, c.quality, c.confidence),
        reverse=True,
    )


def sort_by_composite(candidates: Iterable[Candidate]) -> List[Candidate]:
    return sorted(
        candidates,
        key=lambda c: (c.composite_score, c.quality, c.tps, -c.vram_max_mb),
        reverse=True,
    )


def choose_with_policy(task: str, task_candidates: List[Candidate]) -> Dict[str, Any]:
    policy = TASK_POLICIES.get(task, {"mode": "balanced", "quality_tolerance": QUALITY_TOLERANCE, "roles": ["primary", "fallback"]})
    passing = [c for c in task_candidates if c.passed and c.quality >= PASS_THRESHOLD]
    pool = passing or task_candidates

    if not pool:
        return {"task": task, "note": "no candidates"}

    quality_sorted = sort_by_quality(pool)
    top_quality = quality_sorted[0]
    tolerance = float(policy.get("quality_tolerance", QUALITY_TOLERANCE))
    close_pool = [c for c in pool if (top_quality.quality - c.quality) <= tolerance]

    mode = policy.get("mode", "balanced")
    preferred_model = policy.get("preferred_model")

    primary: Candidate
    if preferred_model:
        preferred = [c for c in pool if c.model == preferred_model]
        primary = preferred[0] if preferred else top_quality
    elif mode == "prefer_quality":
        primary = top_quality
    elif mode == "prefer_efficiency":
        primary = sort_by_efficiency(close_pool)[0]
    else:  # balanced
        primary = sort_by_efficiency(close_pool)[0]

    role_map: Dict[str, str] = {"primary": primary.model}

    # fallback: next best different model by composite score
    remaining = [c for c in sort_by_composite(pool) if c.model != primary.model]
    if "fallback" in policy.get("roles", []) and remaining:
        role_map["fallback"] = remaining[0].model

    # quality: highest quality candidate if different from primary
    if "quality" in policy.get("roles", []):
        if top_quality.model != primary.model:
            role_map["quality"] = top_quality.model
        elif remaining:
            role_map["quality"] = remaining[0].model

    # fast: fastest/most efficient passing candidate if different from primary
    if "fast" in policy.get("roles", []):
        fastest = sort_by_efficiency(pool)[0]
        if fastest.model != primary.model:
            role_map["fast"] = fastest.model
        elif remaining:
            role_map["fast"] = remaining[0].model

    # explanation role for debug: prefer configured explanation model, otherwise top quality non-primary.
    if "explanation" in policy.get("roles", []):
        preferred_explanation = policy.get("explanation_model_preference")
        explanation = None
        if preferred_explanation:
            matches = [c for c in pool if c.model == preferred_explanation]
            if matches:
                explanation = matches[0]
        if explanation is None:
            explanation = top_quality if top_quality.model != primary.model else (remaining[0] if remaining else primary)
        role_map["explanation"] = explanation.model

    return {
        "task": task,
        "recommendation": role_map,
        "selection_policy": {
            "mode": mode,
            "quality_tolerance": tolerance,
            "pass_threshold": PASS_THRESHOLD,
            "preferred_model": preferred_model or None,
            "note": explain_policy(task, mode),
        },
        "selected_primary_metrics": primary.to_row(),
        "top_quality_model": top_quality.model,
        "top_quality_score": round(top_quality.quality, 2),
        "candidate_count": len(task_candidates),
        "passing_candidate_count": len(passing),
        "candidates": [c.to_row() for c in sort_by_composite(task_candidates)],
    }


def explain_policy(task: str, mode: str) -> str:
    if task in {"router", "short_question"}:
        return "輕量前置任務：品質接近時優先 TPS 與 VRAM 效率。"
    if task == "draft_generation":
        return "正式文字任務：primary 優先品質，另提供 fast 模式。"
    if task in {"analysis", "knowledge_refine", "prompt_engineering", "summarization"}:
        return "中階任務：品質差距在容忍範圍內時，優先選效率較好的模型，另保留 quality/fallback。"
    if task == "coding":
        return "程式任務：優先使用 coding 專門模型。"
    if task == "debug":
        return "debug 拆成 implementation 與 explanation：修碼偏 coding 模型，說明/root cause 可用文字能力較強模型。"
    return f"{mode} policy"


def build_recommendations(candidates: List[Candidate]) -> Dict[str, Any]:
    grouped = candidates_by_task(candidates)
    tasks = sorted(grouped.keys())

    by_task = {task: choose_with_policy(task, grouped[task]) for task in tasks}
    routing_recommendations = {
        task: by_task[task].get("recommendation", {})
        for task in tasks
        if by_task[task].get("recommendation")
    }

    return {
        "phase3_version": "3.0-routing-recommendation-generator",
        "generated_from": {
            "phase1_summary_csv": str(PHASE1_SUMMARY_CSV),
            "phase2_summary_csv": str(PHASE2_SUMMARY_CSV),
        },
        "routing_recommendations": routing_recommendations,
        "details_by_task": by_task,
        "global_policy": {
            "quality_tolerance_default": QUALITY_TOLERANCE,
            "pass_threshold": PASS_THRESHOLD,
            "efficiency_score": "0.60 * normalized TPS + 0.25 * inverse normalized VRAM + 0.15 * inverse normalized wall time",
            "composite_score": "0.70 * quality_norm + 0.30 * efficiency_score",
        },
    }


def write_markdown_report(path: Path, recommendations: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Phase 3 Routing Recommendations")
    lines.append("")
    lines.append("## Routing Recommendations")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(recommendations.get("routing_recommendations", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Details by Task")
    lines.append("")

    for task, detail in recommendations.get("details_by_task", {}).items():
        rec = detail.get("recommendation", {})
        primary = rec.get("primary", "")
        selected = detail.get("selected_primary_metrics", {})
        lines.append(f"### {task}")
        lines.append("")
        lines.append(f"- Recommendation: `{json.dumps(rec, ensure_ascii=False)}`")
        if primary:
            lines.append(
                f"- Primary: `{primary}` | quality `{selected.get('quality')}` | "
                f"TPS `{selected.get('tps')}` | VRAM `{selected.get('vram_max_mb')} MB`"
            )
        lines.append(f"- Policy: {detail.get('selection_policy', {}).get('note', '')}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 6. Main
# ============================================================

def main() -> None:
    global PHASE1_SUMMARY_CSV, PHASE2_SUMMARY_CSV

    PHASE1_SUMMARY_CSV = pick_existing_path(PHASE1_SUMMARY_CSV, Path("./phase1_summary.csv"))
    PHASE2_SUMMARY_CSV = pick_existing_path(PHASE2_SUMMARY_CSV, Path("./phase2_summary.csv"))

    print("=== Phase 3 Routing Recommendation Generator ===")
    print(f"Phase 1 summary: {PHASE1_SUMMARY_CSV.resolve()}")
    print(f"Phase 2 summary: {PHASE2_SUMMARY_CSV.resolve()}")
    print(f"Output dir:      {OUTPUT_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    phase1_rows = read_csv_rows(PHASE1_SUMMARY_CSV)
    phase2_rows = read_csv_rows(PHASE2_SUMMARY_CSV)

    candidates = make_candidates(phase1_rows, phase2_rows)
    recommendations = build_recommendations(candidates)

    ROUTING_RECOMMENDATIONS_JSON.write_text(
        json.dumps(recommendations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    candidate_rows = [c.to_row() for c in sorted(candidates, key=lambda c: (c.task, -c.composite_score))]
    write_csv_rows(ROUTING_CANDIDATES_CSV, candidate_rows)
    write_markdown_report(ROUTING_RECOMMENDATIONS_MD, recommendations)

    print("\n--- Phase 3 output complete ---")
    print(f"Recommendations JSON: {ROUTING_RECOMMENDATIONS_JSON.resolve()}")
    print(f"Candidates CSV:       {ROUTING_CANDIDATES_CSV.resolve()}")
    print(f"Markdown report:      {ROUTING_RECOMMENDATIONS_MD.resolve()}")

    print("\n--- Routing Recommendations ---")
    print(json.dumps(recommendations["routing_recommendations"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
