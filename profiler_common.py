"""
Common helpers for Local Model Fit Profiler.

This module intentionally contains no Ollama/OpenAI calls.
It centralizes config loading, task normalization, score handling,
JSON parsing, file naming, and GPU-stat compatibility helpers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_CONFIG_PATH = Path(__file__).with_name("profiler_config.json")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_include_path(base_config_path: Path, include_value: str | Path) -> Path:
    include_path = Path(include_value)
    if include_path.is_absolute():
        return include_path
    return base_config_path.parent / include_path


def _unwrap_include_payload(key: str, payload: Any) -> Any:
    """Support both raw JSON payloads and {key: payload} wrapper files."""
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        if key == "test_plan_task_question" and "task_prompts" in payload:
            return payload["task_prompts"]
    return payload


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load profiler configuration and merge optional external include files.

    Split config supports:
    - task_system_prompts.json via includes.system_prompts
    - profiler_task_prompts.json via includes.test_plan_task_question
    - profiler_test_suite.json via includes.test_suite

    Backward compatibility:
    - config["task_prompts"] is still populated as an alias for
      config["test_plan_task_question"].
    - PROFILER_TASK_PROMPTS remains an alias for
      PROFILER_TEST_PLAN_TASK_QUESTION.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = _load_json(path)
    includes = dict(config.get("includes") or {})

    env_system_prompts = None
    env_test_plan_task_question = None
    env_task_prompts_alias = None
    env_test_suite = None
    try:
        import os

        env_system_prompts = os.getenv("PROFILER_SYSTEM_PROMPTS")
        env_test_plan_task_question = os.getenv("PROFILER_TEST_PLAN_TASK_QUESTION")
        env_task_prompts_alias = os.getenv("PROFILER_TASK_PROMPTS")
        env_test_suite = os.getenv("PROFILER_TEST_SUITE")
    except Exception:
        pass

    include_mapping = {
        "system_prompts": env_system_prompts or includes.get("system_prompts"),
        "test_plan_task_question": (
            env_test_plan_task_question
            or env_task_prompts_alias
            or includes.get("test_plan_task_question")
            or includes.get("task_prompts")
        ),
        "test_suite": env_test_suite or includes.get("test_suite"),
    }

    for key, include_value in include_mapping.items():
        if include_value:
            include_path = _resolve_include_path(path, include_value)
            config[key] = _unwrap_include_payload(key, _load_json(include_path))

    if "test_plan_task_question" not in config and "task_prompts" in config:
        config["test_plan_task_question"] = config["task_prompts"]
    if "task_prompts" not in config and "test_plan_task_question" in config:
        config["task_prompts"] = config["test_plan_task_question"]

    return config


def safe_filename(model_name: str, task_type: str, suffix: str = ".json") -> str:
    normalized_model = (
        str(model_name)
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )
    return f"{normalized_model}_{task_type}{suffix}"


def strip_markdown_fence(text: str) -> str:
    text = str(text or "").strip()
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

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise ValueError(f"Unable to parse JSON from judge output: {text[:500]}")


def ensure_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def known_task_types(config: Dict[str, Any]) -> set[str]:
    task_rubrics = config.get("task_rubrics") or {}
    test_plan_task_question = config.get("test_plan_task_question") or config.get("task_prompts") or {}
    system_prompts = config.get("system_prompts") or {}
    task_weights = config.get("task_weights") or {}
    return set(task_rubrics) | set(test_plan_task_question) | set(system_prompts) | set(task_weights) | {"general"}


def normalize_task_type(task_type: Any, config: Dict[str, Any]) -> str:
    normalized = str(task_type or "general").strip().lower()
    return normalized if normalized in known_task_types(config) else "general"


def get_task_weights(config: Dict[str, Any], task_type: str) -> Dict[str, float]:
    normalized = normalize_task_type(task_type, config)
    weights = config.get("task_weights") or {}
    return weights.get(normalized, weights.get("general", {}))


def get_score_keys(config: Dict[str, Any]) -> List[str]:
    return list(config.get("score_keys") or [
        "accuracy",
        "task_fit",
        "completeness",
        "structure",
        "localization",
        "constraint_following",
    ])


def normalize_score(value: Any) -> Tuple[float, bool]:
    """
    Normalize score to 0~10.

    Returns (normalized_score, scale_fixed). Local judges sometimes output
    0~1 even when asked for 0~10; this converts fractional scores to 0~10.
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0, False

    scale_fixed = False
    if 0.0 <= score <= 1.0:
        score *= 10.0
        scale_fixed = True

    score = max(0.0, min(10.0, score))
    return round(score, 2), scale_fixed


def recompute_weighted_final_score(config: Dict[str, Any], scores: Dict[str, float], task_type: str) -> float:
    weights = get_task_weights(config, task_type)
    total_weight = sum(float(v) for v in weights.values()) or 1.0
    weighted = 0.0
    for key, weight in weights.items():
        weighted += float(scores.get(key, 0.0)) * float(weight)
    return round(weighted / total_weight, 2)


def safe_get_gpu_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    """Support both old and new Phase 1 GPU stat structures."""
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


def get_first(data: Dict[str, Any], *keys: str, default: Any = 0) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default
