"""
Phase 2 - LLM-as-a-Judge for Local Model Fit Profiler

Purpose:
- Read every Phase 1 JSON file under ./phase1_results
- Select a suitable local judge model by task_type
- Grade answer quality with a stable JSON rubric
- Save per-sample grading results under ./phase2_results
- Save aggregate reports for Phase 3 model-fit analysis

Default judge strategy based on current conclusion:
- default judge: gemma3:4b
- coding/debug judge: deepseek-coder:6.7b-instruct
- optional second-pass judge for uncertain cases: mistral:7b-instruct
- qwen3.6:27b is NOT used by default because it is too slow on P15v/P620
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# ============================================================
# 1. Basic Configuration
# ============================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

INPUT_DIR = Path(os.getenv("PHASE1_RESULTS_DIR", "./phase1_results"))
OUTPUT_DIR = Path(os.getenv("PHASE2_RESULTS_DIR", "./phase2_results"))

# Default strategy:
# - Use fast/stable local judge for most tasks.
# - Use code-specialized local judge for coding/debug.
# - Use Mistral only when the first judge says the case is uncertain.
DEFAULT_JUDGE_MODEL = os.getenv("PHASE2_DEFAULT_JUDGE", "gemma3:4b")
CODING_JUDGE_MODEL = os.getenv("PHASE2_CODING_JUDGE", "deepseek-coder:6.7b-instruct")
SECOND_PASS_JUDGE_MODEL = os.getenv("PHASE2_SECOND_PASS_JUDGE", "mistral:7b-instruct")

ENABLE_SECOND_PASS = os.getenv("PHASE2_ENABLE_SECOND_PASS", "true").lower() in {"1", "true", "yes", "y"}
SECOND_PASS_MIN_SCORE = float(os.getenv("PHASE2_SECOND_PASS_MIN_SCORE", "5.0"))
SECOND_PASS_MAX_SCORE = float(os.getenv("PHASE2_SECOND_PASS_MAX_SCORE", "8.0"))
SECOND_PASS_CONFIDENCE_THRESHOLD = float(os.getenv("PHASE2_SECOND_PASS_CONFIDENCE_THRESHOLD", "0.75"))

REQUEST_TIMEOUT_SEC = float(os.getenv("PHASE2_REQUEST_TIMEOUT_SEC", "900"))
SLEEP_BETWEEN_REQUESTS_SEC = float(os.getenv("PHASE2_SLEEP_BETWEEN_REQUESTS_SEC", "2"))

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


# ============================================================
# 2. Task-Specific Rubric Profiles
# ============================================================

TASK_RUBRICS: Dict[str, str] = {
    "router": """
重點評估：
- 是否只做任務分類，而不是回答原問題
- task_type 是否合理
- JSON 格式是否穩定、可被程式解析
- confidence / reason 是否有助於後續 routing
""".strip(),
    "short_question": """
重點評估：
- 是否用簡短篇幅回答核心問題
- 定義是否準確
- 是否避免不必要展開
- 是否適合快速理解
""".strip(),
    "draft_generation": """
重點評估：
- 是否符合指定場景與語氣
- 是否可直接複製使用
- 結構是否清楚
- 是否避免編造未提供的日期、金額、承諾或責任歸屬
""".strip(),
    "analysis": """
重點評估：
- 是否先給結論再展開理由
- 推理是否正確且有層次
- 是否清楚標示假設與不確定性
- 是否提供可落地的下一步或判斷方式
""".strip(),
    "coding": """
重點評估：
- 程式碼是否正確、可執行、符合題意
- 是否有合理型別、錯誤處理與可維護性
- 是否避免不必要套件或過度設計
- 說明是否足以讓使用者理解與使用
""".strip(),
    "debug": """
重點評估：
- 是否找出合理 root cause
- 是否依據錯誤訊息或現象推論，而非亂猜
- 是否提供最小修正與驗證方式
- 是否能沉澱成可重用知識
""".strip(),
    "summarization": """
重點評估：
- 是否忠於原文，沒有加入未提供資訊
- 是否保留關鍵名詞、指令、錯誤訊息、版本號
- 是否整理出核心結論、重點與行動項目
- 是否便於後續搜尋與引用
""".strip(),
    "knowledge_refine": """
重點評估：
- 是否把原始內容抽象化為可長期保存的知識
- 是否移除過度專案化或敏感細節
- 是否萃取問題類型、適用情境、核心原則、常見錯誤與建議做法
- 是否避免把一般知識誤升級成鐵律
""".strip(),
    "prompt_engineering": """
重點評估：
- 是否可直接複製給 AI 工具使用
- 是否明確定義角色、任務、輸入、輸出、限制與驗收標準
- 若是 coding agent prompt，是否包含先讀結構、最小變更、驗證與回報
- 是否符合使用者指定工具場景
""".strip(),
    "general": """
重點評估：
- 是否正確回應使用者主要需求
- 是否清楚、有幫助、沒有過度延伸
- 是否符合繁體中文語境
""".strip(),
}


@dataclass
class JudgeDecision:
    judge_model: str
    reason: str


def choose_primary_judge(task_type: str) -> JudgeDecision:
    """Choose the primary judge model by task_type."""
    normalized = (task_type or "general").strip().lower()
    if normalized in {"coding", "debug"}:
        return JudgeDecision(
            judge_model=CODING_JUDGE_MODEL,
            reason="coding/debug 任務使用程式碼專門 judge",
        )
    return JudgeDecision(
        judge_model=DEFAULT_JUDGE_MODEL,
        reason="一般任務使用預設本地 judge，平衡速度與品質",
    )


# ============================================================
# 3. Prompt & JSON Handling
# ============================================================

JUDGE_SYSTEM_PROMPT = """
你是一位嚴格但公平的 AI 回答品質評測員。

你的任務：
根據使用者原始問題、task_type、模型回答與該任務的評估規準，輸出可被程式解析的 JSON 評分。

評分原則：
1. 只評估模型回答本身，不要重新回答問題。
2. 不要因為回答很長就給高分；重點是是否符合任務。
3. 不要因為回答很短就給低分；short_question 本來就應該短。
4. 若回答包含明顯錯誤、離題、格式不符或編造資訊，應扣分。
5. 若無法確定，降低 confidence，並在 reason 說明。
6. 嚴格輸出 JSON，不要輸出 Markdown，不要包 ```json。

分數定義：
- 9-10：非常適合此 task_type，可直接使用
- 7-8：大致可用，但有小缺點
- 5-6：部分可用，需要修改或複評
- 3-4：明顯不足，不建議使用
- 1-2：錯誤、離題、無法使用

請輸出以下 JSON 格式：
{
  "scores": {
    "accuracy": 0,
    "task_fit": 0,
    "completeness": 0,
    "structure": 0,
    "localization": 0,
    "constraint_following": 0,
    "overall_quality": 0
  },
  "pass": true,
  "confidence": 0.0,
  "reason": "用繁體中文簡短說明主要評分理由",
  "strengths": ["優點1", "優點2"],
  "issues": ["問題1", "問題2"],
  "suggested_improvement": "最重要的一個改善方向"
}
""".strip()


def build_judge_user_prompt(phase1_data: Dict[str, Any]) -> str:
    task_type = str(phase1_data.get("task") or phase1_data.get("task_type") or "general")
    rubric = TASK_RUBRICS.get(task_type, TASK_RUBRICS["general"])
    model_name = phase1_data.get("model", "unknown")
    prompt = phase1_data.get("test_plan_task_question") or phase1_data.get("user_prompt") or phase1_data.get("prompt", "")
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

請依照 system 指示輸出 JSON 評分。
""".strip()


def strip_markdown_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    """Robustly parse a JSON object from common LLM outputs."""
    cleaned = strip_markdown_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: find the largest JSON-looking object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        return json.loads(candidate)

    raise ValueError(f"Unable to parse JSON from judge output: {text[:500]}")


def normalize_grade(raw_grade: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize judge output to a predictable schema."""
    scores = raw_grade.get("scores") or {}

    # Backward compatibility if judge returns quality_score directly.
    if not scores and "quality_score" in raw_grade:
        q = float(raw_grade.get("quality_score", 0))
        scores = {
            "accuracy": q,
            "task_fit": q,
            "completeness": q,
            "structure": q,
            "localization": q,
            "constraint_following": q,
            "overall_quality": q,
        }

    score_keys = [
        "accuracy",
        "task_fit",
        "completeness",
        "structure",
        "localization",
        "constraint_following",
        "overall_quality",
    ]

    normalized_scores: Dict[str, float] = {}
    for key in score_keys:
        try:
            value = float(scores.get(key, 0))
        except (TypeError, ValueError):
            value = 0.0
        normalized_scores[key] = max(0.0, min(10.0, value))

    overall = normalized_scores["overall_quality"]
    confidence = raw_grade.get("confidence", 0.0)
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    confidence_value = max(0.0, min(1.0, confidence_value))

    if "pass" in raw_grade:
        passed = bool(raw_grade["pass"])
    else:
        passed = overall >= 7.0

    return {
        "scores": normalized_scores,
        "pass": passed,
        "confidence": confidence_value,
        "reason": str(raw_grade.get("reason", "")),
        "strengths": ensure_string_list(raw_grade.get("strengths", [])),
        "issues": ensure_string_list(raw_grade.get("issues", [])),
        "suggested_improvement": str(raw_grade.get("suggested_improvement", "")),
    }


def ensure_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


# ============================================================
# 4. Ollama OpenAI-Compatible Call
# ============================================================


def call_judge_model(judge_model: str, phase1_data: Dict[str, Any]) -> Tuple[Dict[str, Any], str, float]:
    """Call a local Ollama judge model and return normalized grade, raw output, elapsed seconds."""
    user_prompt = build_judge_user_prompt(phase1_data)
    start = time.time()
    completion = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        timeout=REQUEST_TIMEOUT_SEC,
    )
    elapsed = time.time() - start
    raw_content = completion.choices[0].message.content or ""
    parsed = extract_json_object(raw_content)
    grade = normalize_grade(parsed)
    return grade, raw_content, elapsed


def needs_second_pass(primary_grade: Dict[str, Any]) -> bool:
    if not ENABLE_SECOND_PASS:
        return False

    overall = float(primary_grade.get("scores", {}).get("overall_quality", 0.0))
    confidence = float(primary_grade.get("confidence", 0.0))

    score_is_ambiguous = SECOND_PASS_MIN_SCORE <= overall <= SECOND_PASS_MAX_SCORE
    confidence_is_low = confidence < SECOND_PASS_CONFIDENCE_THRESHOLD

    return score_is_ambiguous or confidence_is_low


def merge_grades(primary: Dict[str, Any], secondary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge grades. If no secondary, primary is final. If secondary exists, average numeric scores."""
    if not secondary:
        return primary

    merged_scores: Dict[str, float] = {}
    for key, p_val in primary.get("scores", {}).items():
        s_val = secondary.get("scores", {}).get(key, p_val)
        merged_scores[key] = round((float(p_val) + float(s_val)) / 2.0, 2)

    merged_confidence = round(
        (float(primary.get("confidence", 0.0)) + float(secondary.get("confidence", 0.0))) / 2.0,
        2,
    )

    overall = merged_scores.get("overall_quality", 0.0)

    return {
        "scores": merged_scores,
        "pass": overall >= 7.0,
        "confidence": merged_confidence,
        "reason": f"Primary: {primary.get('reason', '')} / Secondary: {secondary.get('reason', '')}",
        "strengths": list(dict.fromkeys(primary.get("strengths", []) + secondary.get("strengths", []))),
        "issues": list(dict.fromkeys(primary.get("issues", []) + secondary.get("issues", []))),
        "suggested_improvement": secondary.get("suggested_improvement") or primary.get("suggested_improvement", ""),
    }


# ============================================================
# 5. Phase 1 Metadata Compatibility Helpers
# ============================================================


def safe_get_gpu_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    """Support both old and new Phase 1 structures."""
    stats = data.get("gpu_stats")
    if isinstance(stats, dict):
        return {
            "vram_max": stats.get("vram_max") or stats.get("vram_max_mb") or stats.get("VRAM_Max") or 0,
            "gpu_util_avg": stats.get("util_avg") or stats.get("gpu_util_avg") or 0,
            "temp_peak": stats.get("temp_peak") or stats.get("temp_peak_c") or 0,
        }

    if isinstance(stats, list) and stats:
        vram_values = [s.get("vram_used_mb", 0) for s in stats if isinstance(s, dict)]
        util_values = [s.get("gpu_util_%", 0) for s in stats if isinstance(s, dict)]
        temp_values = [s.get("temp_c", 0) for s in stats if isinstance(s, dict)]
        return {
            "vram_max": max(vram_values) if vram_values else 0,
            "gpu_util_avg": round(sum(util_values) / len(util_values), 2) if util_values else 0,
            "temp_peak": max(temp_values) if temp_values else 0,
        }

    return {
        "vram_max": data.get("VRAM_Max") or data.get("VRAM_Max_MB") or 0,
        "gpu_util_avg": data.get("GPU_Util_Avg") or 0,
        "temp_peak": data.get("Temp_Peak") or data.get("Temp_Peak_C") or 0,
    }


def make_output_filename(input_path: Path) -> Path:
    return OUTPUT_DIR / f"{input_path.stem}.phase2_judge.json"


# ============================================================
# 6. Main Batch Runner
# ============================================================


def grade_one_file(input_path: Path) -> Dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as f:
        phase1_data = json.load(f)

    task_type = str(phase1_data.get("task") or phase1_data.get("task_type") or "general")
    model_name = str(phase1_data.get("model", "unknown"))

    judge_decision = choose_primary_judge(task_type)
    print(f"\n>>> Phase 2 評分: {input_path.name}")
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
            f"    primary score={primary_grade['scores']['overall_quality']:.1f} "
            f"confidence={primary_grade['confidence']:.2f} elapsed={primary_elapsed:.1f}s"
        )
    except Exception as e:
        errors.append(f"primary judge failed: {type(e).__name__}: {e}")
        print(f"    primary judge failed: {e}")

    if primary_grade and needs_second_pass(primary_grade):
        # Avoid using the same model as secondary judge.
        if SECOND_PASS_JUDGE_MODEL != judge_decision.judge_model:
            try:
                print(f"    second pass triggered -> judge={SECOND_PASS_JUDGE_MODEL}")
                secondary_grade, raw_secondary, secondary_elapsed = call_judge_model(
                    SECOND_PASS_JUDGE_MODEL,
                    phase1_data,
                )
                print(
                    f"    secondary score={secondary_grade['scores']['overall_quality']:.1f} "
                    f"confidence={secondary_grade['confidence']:.2f} elapsed={secondary_elapsed:.1f}s"
                )
            except Exception as e:
                errors.append(f"secondary judge failed: {type(e).__name__}: {e}")
                print(f"    secondary judge failed: {e}")

    final_grade = merge_grades(primary_grade, secondary_grade) if primary_grade else None
    gpu_stats = safe_get_gpu_stats(phase1_data)

    result = {
        "source_file": str(input_path),
        "graded_at": datetime.now().isoformat(timespec="seconds"),
        "phase1": {
            "model": model_name,
            "task": task_type,
            "test_plan_task_question": phase1_data.get("test_plan_task_question") or phase1_data.get("user_prompt") or phase1_data.get("prompt", ""),
            "prompt": phase1_data.get("test_plan_task_question") or phase1_data.get("user_prompt") or phase1_data.get("prompt", ""),  # backward-compatible alias
            "success": phase1_data.get("success", None),
            "tps": phase1_data.get("tps", phase1_data.get("TPS", 0)),
            "total_duration": phase1_data.get("total_duration", phase1_data.get("Total_Sec", 0)),
            "load_duration": phase1_data.get("load_duration", phase1_data.get("Load_Sec", 0)),
            "eval_duration": phase1_data.get("eval_duration", phase1_data.get("Eval_Sec", 0)),
            "wall_time": phase1_data.get("wall_time", phase1_data.get("Wall_Sec", 0)),
            "gpu_stats": gpu_stats,
        },
        "judge": {
            "primary_model": judge_decision.judge_model,
            "primary_reason": judge_decision.reason,
            "primary_elapsed_sec": round(primary_elapsed, 2),
            "secondary_enabled": ENABLE_SECOND_PASS,
            "secondary_model": SECOND_PASS_JUDGE_MODEL if secondary_grade else None,
            "secondary_elapsed_sec": round(secondary_elapsed, 2),
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

        rows.append(
            {
                "source_file": Path(r.get("source_file", "")).name,
                "model": phase1.get("model", ""),
                "task": phase1.get("task", ""),
                "primary_judge": judge.get("primary_model", ""),
                "secondary_judge": judge.get("secondary_model", ""),
                "quality_score": scores.get("overall_quality", 0),
                "accuracy": scores.get("accuracy", 0),
                "task_fit": scores.get("task_fit", 0),
                "completeness": scores.get("completeness", 0),
                "structure": scores.get("structure", 0),
                "localization": scores.get("localization", 0),
                "constraint_following": scores.get("constraint_following", 0),
                "pass": final.get("pass", False),
                "confidence": final.get("confidence", 0),
                "tps": phase1.get("tps", 0),
                "vram_max_mb": gpu_stats.get("vram_max", 0),
                "gpu_util_avg": gpu_stats.get("gpu_util_avg", 0),
                "temp_peak_c": gpu_stats.get("temp_peak", 0),
                "load_sec": phase1.get("load_duration", 0),
                "wall_sec": phase1.get("wall_time", 0),
                "reason": final.get("reason", ""),
                "suggested_improvement": final.get("suggested_improvement", ""),
                "errors": " | ".join(judge.get("errors", [])),
            }
        )
    return rows


def write_summary_files(results: List[Dict[str, Any]]) -> None:
    rows = build_summary_rows(results)

    summary_json = OUTPUT_DIR / "phase2_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    summary_csv = OUTPUT_DIR / "phase2_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "source_file",
        "model",
        "task",
        "quality_score",
        "tps",
        "reason",
    ]
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Lightweight leaderboard by task + model.
    leaderboard: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        task = str(row.get("task", "general"))
        leaderboard.setdefault(task, []).append(row)
    for task, task_rows in leaderboard.items():
        task_rows.sort(
            key=lambda x: (
                float(x.get("quality_score") or 0),
                float(x.get("tps") or 0),
                -float(x.get("vram_max_mb") or 0),
            ),
            reverse=True,
        )

    leaderboard_json = OUTPUT_DIR / "phase2_leaderboard_by_task.json"
    with leaderboard_json.open("w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)

    print("\n--- Phase 2 輸出完成 ---")
    print(f"Per-file results: {OUTPUT_DIR.resolve()}")
    print(f"Summary JSON:     {summary_json.resolve()}")
    print(f"Summary CSV:      {summary_csv.resolve()}")
    print(f"Leaderboard:      {leaderboard_json.resolve()}")

    if rows:
        print("\n--- 簡短總表 ---")
        for row in sorted(rows, key=lambda x: (x["task"], -float(x["quality_score"] or 0))):
            print(
                f"task={row['task']:<18} model={row['model']:<30} "
                f"score={float(row['quality_score'] or 0):>4.1f} "
                f"tps={float(row['tps'] or 0):>6.2f} "
                f"judge={row['primary_judge']}"
            )


def run_phase2_grading() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(Path(p) for p in glob.glob(str(INPUT_DIR / "*.json")))
    if not input_files:
        print(f"找不到 Phase 1 JSON：{INPUT_DIR.resolve()}/*.json")
        print("請先確認 phase1-profiler-batch.py 已產生 ./phase1_results/*.json")
        return

    print("=== Phase 2 LLM-as-a-Judge 開始 ===")
    print(f"Input dir:  {INPUT_DIR.resolve()}")
    print(f"Output dir: {OUTPUT_DIR.resolve()}")
    print(f"Default judge: {DEFAULT_JUDGE_MODEL}")
    print(f"Coding judge:  {CODING_JUDGE_MODEL}")
    print(f"Second pass:   {ENABLE_SECOND_PASS} ({SECOND_PASS_JUDGE_MODEL})")
    print(f"Files found:   {len(input_files)}")

    results: List[Dict[str, Any]] = []
    for input_path in input_files:
        try:
            result = grade_one_file(input_path)
            results.append(result)
        except Exception as e:
            print(f"\n!!! 無法處理 {input_path.name}: {type(e).__name__}: {e}")

        if SLEEP_BETWEEN_REQUESTS_SEC > 0:
            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

    write_summary_files(results)


if __name__ == "__main__":
    run_phase2_grading()
