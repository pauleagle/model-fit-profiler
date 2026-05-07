"""
Phase 2.1 - LLM-as-a-Judge, refactored.

Responsibilities kept here:
- read Phase 1 JSON files
- call local judge models through Ollama OpenAI-compatible API
- perform optional second pass
- write per-file grading results and aggregate reports

Moved to external config/common files:
- judge models
- task rubrics
- score keys / task weights
- judge system prompts
- score normalization / JSON parsing helpers
"""

from __future__ import annotations

import csv
import glob
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from profiler_common import (
    ensure_string_list,
    extract_json_object,
    get_first,
    get_score_keys,
    get_task_weights,
    load_config,
    normalize_score,
    normalize_task_type,
    recompute_weighted_final_score,
    safe_get_gpu_stats,
)

CONFIG_PATH = Path(os.getenv("PROFILER_CONFIG", "profiler_config.json"))
CONFIG = load_config(CONFIG_PATH)

OLLAMA_BASE_URL = os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

INPUT_DIR = Path(os.getenv("PHASE1_RESULTS_DIR", "./phase1_results"))
OUTPUT_DIR = Path(os.getenv("PHASE2_RESULTS_DIR", "./phase2_results"))

JUDGE_MODELS = CONFIG.get("judge_models", {})
DEFAULT_JUDGE_MODEL = os.getenv("PHASE2_DEFAULT_JUDGE", JUDGE_MODELS.get("default", "gemma3:4b"))
CODING_JUDGE_MODEL = os.getenv("PHASE2_CODING_JUDGE", JUDGE_MODELS.get("coding", "deepseek-coder:6.7b-instruct"))
SECOND_PASS_JUDGE_MODEL = os.getenv("PHASE2_SECOND_PASS_JUDGE", JUDGE_MODELS.get("second_pass", "mistral:7b-instruct"))

ENABLE_SECOND_PASS = os.getenv("PHASE2_ENABLE_SECOND_PASS", "true").lower() in {"1", "true", "yes", "y"}
SECOND_PASS_MIN_SCORE = float(os.getenv("PHASE2_SECOND_PASS_MIN_SCORE", "5.0"))
SECOND_PASS_MAX_SCORE = float(os.getenv("PHASE2_SECOND_PASS_MAX_SCORE", "8.0"))
SECOND_PASS_CONFIDENCE_THRESHOLD = float(os.getenv("PHASE2_SECOND_PASS_CONFIDENCE_THRESHOLD", "0.75"))

REQUEST_TIMEOUT_SEC = float(os.getenv("PHASE2_REQUEST_TIMEOUT_SEC", "900"))
SLEEP_BETWEEN_REQUESTS_SEC = float(os.getenv("PHASE2_SLEEP_BETWEEN_REQUESTS_SEC", "2"))
PASS_THRESHOLD = float(os.getenv("PHASE2_PASS_THRESHOLD", "7.0"))

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


@dataclass
class JudgeDecision:
    judge_model: str
    reason: str


def choose_primary_judge(task_type: str) -> JudgeDecision:
    normalized = normalize_task_type(task_type, CONFIG)
    if normalized in {"coding", "debug"}:
        return JudgeDecision(CODING_JUDGE_MODEL, "coding/debug 任務使用程式碼專門 judge")
    return JudgeDecision(DEFAULT_JUDGE_MODEL, "一般任務使用預設本地 judge，平衡速度與品質")


def get_judge_system_prompt(task_type: str) -> str:
    prompts = CONFIG.get("judge_system_prompts", {})
    if normalize_task_type(task_type, CONFIG) == "router":
        return prompts.get("router", prompts.get("base", ""))
    return prompts.get("base", "")


def build_judge_user_prompt(phase1_data: Dict[str, Any]) -> str:
    task_type = normalize_task_type(phase1_data.get("task") or phase1_data.get("task_type"), CONFIG)
    rubric = (CONFIG.get("task_rubrics") or {}).get(task_type, (CONFIG.get("task_rubrics") or {}).get("general", ""))
    weights = get_task_weights(CONFIG, task_type)
    model_name = phase1_data.get("model", "unknown")
    prompt = phase1_data.get("user_prompt") or phase1_data.get("prompt", "")
    response = phase1_data.get("response", "")

    return f"""
[task_type]
{task_type}

[被評估模型]
{model_name}

[原始問題 / 測試 prompt]
{prompt}

[模型回答]
{response}

[此 task_type 的評估規準]
{rubric}

[Phase 2.1 本地重算 weighted_final_score 的權重]
{json.dumps(weights, ensure_ascii=False, indent=2)}

請依照 system 指示輸出 JSON 評分。注意：你的所有 score 都必須是 0 到 10，不是 0 到 1。
""".strip()


def normalize_grade(raw_grade: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    score_keys = get_score_keys(CONFIG)
    scores = raw_grade.get("scores") or {}

    if not scores and "quality_score" in raw_grade:
        q = raw_grade.get("quality_score", 0)
        scores = {key: q for key in score_keys}
        scores["overall_quality"] = q

    normalized_scores: Dict[str, float] = {}
    scale_fixed_fields: List[str] = []

    for key in score_keys + ["overall_quality"]:
        value, fixed = normalize_score(scores.get(key, 0))
        normalized_scores[key] = value
        if fixed:
            scale_fixed_fields.append(key)

    weighted_final_score = recompute_weighted_final_score(CONFIG, normalized_scores, task_type)

    try:
        confidence_value = float(raw_grade.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence_value = 0.0
    confidence_value = max(0.0, min(1.0, confidence_value))

    judge_pass = bool(raw_grade.get("pass", weighted_final_score >= PASS_THRESHOLD))
    final_pass = weighted_final_score >= PASS_THRESHOLD

    return {
        "scores": normalized_scores,
        "raw_overall_quality": normalized_scores.get("overall_quality", 0.0),
        "weighted_final_score": weighted_final_score,
        "judge_pass": judge_pass,
        "pass": final_pass,
        "confidence": round(confidence_value, 2),
        "reason": str(raw_grade.get("reason", "")),
        "strengths": ensure_string_list(raw_grade.get("strengths", [])),
        "issues": ensure_string_list(raw_grade.get("issues", [])),
        "suggested_improvement": str(raw_grade.get("suggested_improvement", "")),
        "score_scale_fixed": bool(scale_fixed_fields),
        "score_scale_fixed_fields": scale_fixed_fields,
        "judge_prompt_type": "router" if normalize_task_type(task_type, CONFIG) == "router" else "task_aware",
    }


def call_judge_model(judge_model: str, phase1_data: Dict[str, Any]) -> Tuple[Dict[str, Any], str, float]:
    task_type = normalize_task_type(phase1_data.get("task") or phase1_data.get("task_type"), CONFIG)
    system_prompt = get_judge_system_prompt(task_type)
    user_prompt = build_judge_user_prompt(phase1_data)

    start = time.time()
    completion = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        timeout=REQUEST_TIMEOUT_SEC,
    )
    elapsed = time.time() - start
    raw_content = completion.choices[0].message.content or ""
    parsed = extract_json_object(raw_content)
    grade = normalize_grade(parsed, task_type)
    return grade, raw_content, elapsed


def needs_second_pass(primary_grade: Dict[str, Any]) -> bool:
    if not ENABLE_SECOND_PASS:
        return False
    final_score = float(primary_grade.get("weighted_final_score", 0.0))
    confidence = float(primary_grade.get("confidence", 0.0))
    return (SECOND_PASS_MIN_SCORE <= final_score <= SECOND_PASS_MAX_SCORE) or (confidence < SECOND_PASS_CONFIDENCE_THRESHOLD)


def merge_grades(primary: Dict[str, Any], secondary: Optional[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
    if not secondary:
        return primary

    score_keys = get_score_keys(CONFIG)
    merged_scores: Dict[str, float] = {}
    for key in score_keys + ["overall_quality"]:
        p_val = primary.get("scores", {}).get(key, 0.0)
        s_val = secondary.get("scores", {}).get(key, p_val)
        merged_scores[key] = round((float(p_val) + float(s_val)) / 2.0, 2)

    weighted_final_score = recompute_weighted_final_score(CONFIG, merged_scores, task_type)
    merged_confidence = round((float(primary.get("confidence", 0.0)) + float(secondary.get("confidence", 0.0))) / 2.0, 2)
    scale_fixed_fields = sorted(set(primary.get("score_scale_fixed_fields", [])) | set(secondary.get("score_scale_fixed_fields", [])))

    return {
        "scores": merged_scores,
        "raw_overall_quality": merged_scores.get("overall_quality", 0.0),
        "weighted_final_score": weighted_final_score,
        "judge_pass": bool(primary.get("judge_pass", False)) and bool(secondary.get("judge_pass", False)),
        "pass": weighted_final_score >= PASS_THRESHOLD,
        "confidence": merged_confidence,
        "reason": f"Primary: {primary.get('reason', '')} / Secondary: {secondary.get('reason', '')}",
        "strengths": list(dict.fromkeys(primary.get("strengths", []) + secondary.get("strengths", []))),
        "issues": list(dict.fromkeys(primary.get("issues", []) + secondary.get("issues", []))),
        "suggested_improvement": secondary.get("suggested_improvement") or primary.get("suggested_improvement", ""),
        "score_scale_fixed": bool(scale_fixed_fields),
        "score_scale_fixed_fields": scale_fixed_fields,
        "judge_prompt_type": primary.get("judge_prompt_type", "task_aware"),
    }


def make_output_filename(input_path: Path) -> Path:
    return OUTPUT_DIR / f"{input_path.stem}.phase2_judge.json"


def grade_one_file(input_path: Path) -> Dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as f:
        phase1_data = json.load(f)

    task_type = normalize_task_type(phase1_data.get("task") or phase1_data.get("task_type"), CONFIG)
    model_name = str(phase1_data.get("model", "unknown"))
    judge_decision = choose_primary_judge(task_type)

    print(f"\n>>> Phase 2.1 評分: {input_path.name}")
    print(f"    model={model_name} | task={task_type} | judge={judge_decision.judge_model}")

    primary_grade: Optional[Dict[str, Any]] = None
    secondary_grade: Optional[Dict[str, Any]] = None
    raw_primary = ""
    raw_secondary = ""
    primary_elapsed = 0.0
    secondary_elapsed = 0.0
    errors: List[str] = []

    try:
        primary_grade, raw_primary, primary_elapsed = call_judge_model(judge_decision.judge_model, phase1_data)
        print(
            f"    primary weighted={primary_grade['weighted_final_score']:.2f} "
            f"raw_overall={primary_grade['raw_overall_quality']:.2f} "
            f"confidence={primary_grade['confidence']:.2f} "
            f"scale_fixed={primary_grade['score_scale_fixed']} "
            f"elapsed={primary_elapsed:.1f}s"
        )
    except Exception as e:
        errors.append(f"primary judge failed: {type(e).__name__}: {e}")
        print(f"    primary judge failed: {e}")

    if primary_grade and needs_second_pass(primary_grade) and SECOND_PASS_JUDGE_MODEL != judge_decision.judge_model:
        try:
            print(f"    second pass triggered -> judge={SECOND_PASS_JUDGE_MODEL}")
            secondary_grade, raw_secondary, secondary_elapsed = call_judge_model(SECOND_PASS_JUDGE_MODEL, phase1_data)
            print(
                f"    secondary weighted={secondary_grade['weighted_final_score']:.2f} "
                f"raw_overall={secondary_grade['raw_overall_quality']:.2f} "
                f"confidence={secondary_grade['confidence']:.2f} "
                f"scale_fixed={secondary_grade['score_scale_fixed']} "
                f"elapsed={secondary_elapsed:.1f}s"
            )
        except Exception as e:
            errors.append(f"secondary judge failed: {type(e).__name__}: {e}")
            print(f"    secondary judge failed: {e}")

    final_grade = merge_grades(primary_grade, secondary_grade, task_type) if primary_grade else None
    gpu_stats = safe_get_gpu_stats(phase1_data)

    result = {
        "source_file": str(input_path),
        "graded_at": datetime.now().isoformat(timespec="seconds"),
        "phase2_version": "2.1-refactored-config-common",
        "phase1": {
            "model": model_name,
            "task": task_type,
            "prompt": phase1_data.get("user_prompt") or phase1_data.get("prompt", ""),
            "success": phase1_data.get("success", None),
            "tps": phase1_data.get("tps", phase1_data.get("TPS", 0)),
            "total_duration": get_first(phase1_data, "total_duration", "Total_Sec"),
            "load_duration": get_first(phase1_data, "load_duration", "Load_Sec"),
            "eval_duration": get_first(phase1_data, "eval_duration", "Eval_Sec"),
            "wall_time": get_first(phase1_data, "wall_time", "Wall_Sec"),
            "gpu_stats": gpu_stats,
        },
        "judge": {
            "primary_model": judge_decision.judge_model,
            "primary_reason": judge_decision.reason,
            "primary_elapsed_sec": round(primary_elapsed, 2),
            "secondary_enabled": ENABLE_SECOND_PASS,
            "secondary_model": SECOND_PASS_JUDGE_MODEL if secondary_grade else None,
            "secondary_elapsed_sec": round(secondary_elapsed, 2),
            "pass_threshold": PASS_THRESHOLD,
            "task_weights": get_task_weights(CONFIG, task_type),
            "primary_grade": primary_grade,
            "secondary_grade": secondary_grade,
            "final_grade": final_grade,
            "raw_primary_output": raw_primary,
            "raw_secondary_output": raw_secondary,
            "errors": errors,
        },
    }

    out_path = make_output_filename(input_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def build_summary_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in results:
        phase1 = r.get("phase1", {})
        judge = r.get("judge", {})
        final = judge.get("final_grade") or {}
        scores = final.get("scores") or {}
        gpu_stats = phase1.get("gpu_stats") or {}
        rows.append({
            "source_file": Path(r.get("source_file", "")).name,
            "model": phase1.get("model", ""),
            "task": phase1.get("task", ""),
            "primary_judge": judge.get("primary_model", ""),
            "secondary_judge": judge.get("secondary_model", ""),
            "raw_quality_score": final.get("raw_overall_quality", scores.get("overall_quality", 0)),
            "weighted_final_score": final.get("weighted_final_score", 0),
            "quality_score": final.get("weighted_final_score", 0),
            "accuracy": scores.get("accuracy", 0),
            "task_fit": scores.get("task_fit", 0),
            "completeness": scores.get("completeness", 0),
            "structure": scores.get("structure", 0),
            "localization": scores.get("localization", 0),
            "constraint_following": scores.get("constraint_following", 0),
            "pass": final.get("pass", False),
            "judge_pass": final.get("judge_pass", False),
            "confidence": final.get("confidence", 0),
            "score_scale_fixed": final.get("score_scale_fixed", False),
            "score_scale_fixed_fields": ",".join(final.get("score_scale_fixed_fields", [])),
            "judge_prompt_type": final.get("judge_prompt_type", ""),
            "tps": phase1.get("tps", 0),
            "vram_max_mb": gpu_stats.get("vram_max", 0),
            "gpu_util_avg": gpu_stats.get("gpu_util_avg", 0),
            "temp_peak_c": gpu_stats.get("temp_peak", 0),
            "load_sec": phase1.get("load_duration", 0),
            "wall_sec": phase1.get("wall_time", 0),
            "reason": final.get("reason", ""),
            "suggested_improvement": final.get("suggested_improvement", ""),
            "errors": " | ".join(judge.get("errors", [])),
        })
    return rows


def build_leaderboard(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    leaderboard: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        task = str(row.get("task", "general"))
        leaderboard.setdefault(task, []).append(row)
    for task_rows in leaderboard.values():
        task_rows.sort(
            key=lambda x: (
                float(x.get("weighted_final_score") or 0),
                float(x.get("tps") or 0),
                -float(x.get("vram_max_mb") or 0),
            ),
            reverse=True,
        )
    return leaderboard


def write_summary_files(results: List[Dict[str, Any]]) -> None:
    rows = build_summary_rows(results)
    summary_json = OUTPUT_DIR / "phase2_summary.json"
    summary_csv = OUTPUT_DIR / "phase2_summary.csv"
    leaderboard_json = OUTPUT_DIR / "phase2_leaderboard_by_task.json"
    recommendations_json = OUTPUT_DIR / "phase2_recommendations.json"

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = list(rows[0].keys()) if rows else ["source_file", "model", "task", "weighted_final_score", "tps", "reason"]
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    leaderboard = build_leaderboard(rows)
    with leaderboard_json.open("w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)

    recommendations: Dict[str, Any] = {}
    for task, task_rows in leaderboard.items():
        passing = [r for r in task_rows if r.get("pass")]
        candidates = passing or task_rows
        top = candidates[0] if candidates else None
        if top:
            recommendations[task] = {
                "recommended_model": top.get("model"),
                "weighted_final_score": top.get("weighted_final_score"),
                "tps": top.get("tps"),
                "vram_max_mb": top.get("vram_max_mb"),
                "note": "依 weighted_final_score 優先，其次 TPS 與 VRAM 排序；router 類已用專用 judge prompt。",
            }

    with recommendations_json.open("w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)

    print("\n--- Phase 2.1 輸出完成 ---")
    print(f"Per-file results: {OUTPUT_DIR.resolve()}")
    print(f"Summary JSON:     {summary_json.resolve()}")
    print(f"Summary CSV:      {summary_csv.resolve()}")
    print(f"Leaderboard:      {leaderboard_json.resolve()}")
    print(f"Recommendations:  {recommendations_json.resolve()}")

    if rows:
        print("\n--- 簡短總表 ---")
        for row in sorted(rows, key=lambda x: (x["task"], -float(x["weighted_final_score"] or 0))):
            scale_note = " scale-fixed" if row.get("score_scale_fixed") else ""
            print(
                f"task={row['task']:<18} model={row['model']:<30} "
                f"weighted={float(row['weighted_final_score'] or 0):>4.1f} "
                f"raw={float(row['raw_quality_score'] or 0):>4.1f} "
                f"tps={float(row['tps'] or 0):>6.2f} "
                f"judge={row['primary_judge']}{scale_note}"
            )


def run_phase2_grading() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_files = sorted(Path(p) for p in glob.glob(str(INPUT_DIR / "*.json")))
    if not input_files:
        print(f"找不到 Phase 1 JSON：{INPUT_DIR.resolve()}/*.json")
        print("請先確認 phase1-profiler_batch_refactored.py 已產生 Phase 1 JSON。")
        return

    print("=== Phase 2.1 LLM-as-a-Judge 開始 ===")
    print(f"Config:     {CONFIG_PATH.resolve()}")
    print(f"Input dir:  {INPUT_DIR.resolve()}")
    print(f"Output dir: {OUTPUT_DIR.resolve()}")
    print(f"Default judge: {DEFAULT_JUDGE_MODEL}")
    print(f"Coding judge:  {CODING_JUDGE_MODEL}")
    print(f"Second pass:   {ENABLE_SECOND_PASS} ({SECOND_PASS_JUDGE_MODEL})")
    print(f"Pass threshold: {PASS_THRESHOLD}")
    print(f"Files found:   {len(input_files)}")

    results: List[Dict[str, Any]] = []
    for input_path in input_files:
        try:
            results.append(grade_one_file(input_path))
        except Exception as e:
            print(f"\n!!! 無法處理 {input_path.name}: {type(e).__name__}: {e}")
        if SLEEP_BETWEEN_REQUESTS_SEC > 0:
            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

    write_summary_files(results)


if __name__ == "__main__":
    run_phase2_grading()
