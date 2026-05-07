"""
Common helpers for Local Model Fit Profiler.

This module intentionally contains no Ollama/OpenAI calls.
It centralizes config loading, include merging, task normalization,
score handling, JSON parsing, file naming, and GPU-stat compatibility helpers.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_CONFIG_PATH = Path(__file__).with_name("profiler_config.json")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_include_path(base_dir: Path, env_name: str, configured_path: str | None) -> Path | None:
    override = os.getenv(env_name)
    raw = override or configured_path
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _merge_include(config: Dict[str, Any], key: str, include_data: Any) -> None:
    """
    Include file can be either:
    - {"task_prompts": {...}}
    - raw {...} for task_prompts
    - {"test_suite": [...]}
    - raw [...] for test_suite
    """
    if isinstance(include_data, dict) and key in include_data:
        config[key] = include_data[key]
    else:
        config[key] = include_data


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load profiler_config.json and merge optional include files.

    Supported includes in profiler_config.json:
        "includes": {
          "task_prompts": "profiler_task_prompts.json",
          "test_suite": "profiler_test_suite.json"
        }

    Environment overrides:
        PROFILER_TASK_PROMPTS=/path/to/prompts.json
        PROFILER_TEST_SUITE=/path/to/suite.json
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = _load_json(path)
    base_dir = path.parent
    includes = config.get("includes") or {}

    task_prompts_path = _resolve_include_path(
        base_dir,
        "PROFILER_TASK_PROMPTS",
        includes.get("task_prompts"),
    )
    if task_prompts_path:
        _merge_include(config, "task_prompts", _load_json(task_prompts_path))

    test_suite_path = _resolve_include_path(
        base_dir,
        "PROFILER_TEST_SUITE",
        includes.get("test_suite"),
    )
    if test_suite_path:
        _merge_include(config, "test_suite", _load_json(test_suite_path))

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
    task_prompts = config.get("task_prompts") or {}
    system_prompts = config.get("system_prompts") or {}
    task_weights = config.get("task_weights") or {}
    return set(task_rubrics) | set(task_prompts) | set(system_prompts) | set(task_weights) | {"general"}


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
