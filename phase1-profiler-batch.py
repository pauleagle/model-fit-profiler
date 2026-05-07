"""
Model Fit Profiler - Phase 1 Batch Runner, refactored.

Responsibilities kept here:
- run Ollama generation tests
- monitor GPU stats
- write per-test JSON and summary CSV

Moved to external config/common files:
- task system prompts
- test-plan task questions
- task params
- test suite
- safe filename/config helpers
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from profiler_common import load_config, safe_filename

CONFIG_PATH = Path(os.getenv("PROFILER_CONFIG", "profiler_config.json"))
CONFIG = load_config(CONFIG_PATH)

OLLAMA_URL = os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
OUTPUT_DIR = Path(os.getenv("PHASE1_RESULTS_DIR", "./phase1_results"))
_summary_csv_env = os.getenv("PHASE1_SUMMARY_CSV")
SUMMARY_CSV = Path(_summary_csv_env) if _summary_csv_env else OUTPUT_DIR / "phase1_summary.csv"
REQUEST_TIMEOUT_SEC = float(os.getenv("PHASE1_REQUEST_TIMEOUT_SEC", "600"))
GPU_SAMPLE_INTERVAL_SEC = float(os.getenv("GPU_SAMPLE_INTERVAL_SEC", "0.5"))
COOLDOWN_SEC = float(os.getenv("PHASE1_COOLDOWN_SEC", "10"))
CLEAR_VRAM_WAIT_SEC = float(os.getenv("CLEAR_VRAM_WAIT_SEC", "5"))

SYSTEM_PROMPTS: Dict[str, str] = CONFIG["system_prompts"]
TEST_PLAN_TASK_QUESTION: Dict[str, str] = CONFIG["test_plan_task_question"]
TASK_PARAMS: Dict[str, Dict[str, Any]] = CONFIG["task_params"]
TEST_SUITE: List[Dict[str, Any]] = CONFIG["test_suite"]

gpu_metrics: List[Dict[str, Any]] = []
stop_monitoring = False


def get_gpu_status() -> Optional[Dict[str, Any]]:
    """Get GPU status. Tolerates N/A values from Quadro P620 on Windows WDDM."""
    try:
        cmd = (
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu "
            "--format=csv,noheader,nounits"
        )
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
        if not result:
            return None

        raw_vals = result.split(",")
        vals = [float(v.strip()) if v.strip().upper() != "[N/A]" else 0.0 for v in raw_vals]
        return {
            "time": datetime.now().strftime("%H:%M:%S"),
            "gpu_util_%": vals[0],
            "vram_used_mb": vals[1],
            "temp_c": vals[2],
        }
    except Exception:
        return None


def monitor_loop() -> None:
    global stop_monitoring, gpu_metrics
    while not stop_monitoring:
        status = get_gpu_status()
        if status:
            gpu_metrics.append(status)
        time.sleep(GPU_SAMPLE_INTERVAL_SEC)


def clear_vram() -> None:
    """Unload current Ollama model before each test to keep load_duration comparable."""
    print("清理 VRAM / 卸載 Ollama 模型中...")
    try:
        httpx.post(OLLAMA_URL, json={"model": "none", "keep_alive": 0}, timeout=10)
        time.sleep(CLEAR_VRAM_WAIT_SEC)
    except Exception:
        pass


def build_prompt(task_type: str) -> str:
    system_prompt = SYSTEM_PROMPTS[task_type]
    user_prompt = TEST_PLAN_TASK_QUESTION[task_type]
    return f"""<system>
{system_prompt}
</system>

<user>
{user_prompt}
</user>

請根據 system 指示回答。
""".strip()


def run_single_test(model_name: str, task_type: str) -> Dict[str, Any]:
    global stop_monitoring, gpu_metrics

    if task_type not in SYSTEM_PROMPTS:
        raise ValueError(f"Missing system prompt for task_type: {task_type}")
    if task_type not in TEST_PLAN_TASK_QUESTION:
        raise ValueError(f"Missing test-plan task question for task_type: {task_type}")

    gpu_metrics = []
    stop_monitoring = False
    monitor_thread = threading.Thread(target=monitor_loop)
    monitor_thread.start()

    params = TASK_PARAMS.get(task_type, {})
    composed_prompt = build_prompt(task_type)

    result_data: Dict[str, Any] = {
        "model": model_name,
        "task": task_type,
        "system_prompt": SYSTEM_PROMPTS[task_type],
        "test_plan_task_question": TEST_PLAN_TASK_QUESTION[task_type],
        "user_prompt": TEST_PLAN_TASK_QUESTION[task_type],  # backward-compatible alias
        "composed_prompt": composed_prompt,
        "params": params,
        "success": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        payload = {
            "model": model_name,
            "prompt": composed_prompt,
            "stream": False,
            "keep_alive": 0,
            "options": {
                "temperature": params.get("temperature", 0.2),
                "num_predict": params.get("num_predict", 1024),
            },
        }

        with httpx.Client(timeout=REQUEST_TIMEOUT_SEC) as client:
            start_wall_time = time.time()
            response = client.post(OLLAMA_URL, json=payload)
            end_wall_time = time.time()

        response.raise_for_status()
        res_json = response.json()

        total_duration_sec = res_json.get("total_duration", 0) / 1e9
        load_duration_sec = res_json.get("load_duration", 0) / 1e9
        prompt_eval_count = res_json.get("prompt_eval_count", 0)
        eval_count = res_json.get("eval_count", 0)
        eval_duration_sec = res_json.get("eval_duration", 0) / 1e9

        if eval_duration_sec > 0 and eval_count > 0:
            tps = eval_count / eval_duration_sec
        elif eval_count > 0 and total_duration_sec > load_duration_sec:
            tps = eval_count / (total_duration_sec - load_duration_sec)
        else:
            tps = 0

        result_data.update({
            "success": True,
            "response": res_json.get("response"),
            "done_reason": res_json.get("done_reason"),
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "tps": round(tps, 2),
            "total_duration": round(total_duration_sec, 2),
            "load_duration": round(load_duration_sec, 2),
            "eval_duration": round(eval_duration_sec, 2),
            "wall_time": round(end_wall_time - start_wall_time, 2),
        })
        print(
            f"完成: {model_name} | {task_type} | "
            f"{round(tps, 2)} t/s | load {round(load_duration_sec, 2)}s | tokens {eval_count}"
        )

    except Exception as e:
        print(f"測試異常: {model_name} | {task_type} | {e}")
        result_data.update({"error": str(e)})

    finally:
        stop_monitoring = True
        monitor_thread.join()

        if gpu_metrics:
            df = pd.DataFrame(gpu_metrics)
            result_data["gpu_stats"] = {
                "util_avg": round(df["gpu_util_%"].mean(), 2),
                "util_peak": round(df["gpu_util_%"].max(), 2),
                "vram_avg": int(df["vram_used_mb"].mean()),
                "vram_max": int(df["vram_used_mb"].max()),
                "temp_avg": round(df["temp_c"].mean(), 2),
                "temp_peak": int(df["temp_c"].max()),
                "samples": len(df),
            }
        else:
            result_data["gpu_stats"] = {}

    return result_data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_summary: List[Dict[str, Any]] = []

    print("=== Model Fit Profiler: Phase 1 task_type batch test start ===")
    print(f"Config:     {CONFIG_PATH.resolve()}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Output dir: {OUTPUT_DIR}")

    for entry in TEST_SUITE:
        model = entry["model"]
        tasks = entry["tasks"]

        for task in tasks:
            print("\n" + ">" * 60)
            print(f"測試目標: {model} | task_type: {task}")
            print(">" * 60)

            clear_vram()
            test_result = run_single_test(model, task)

            output_path = OUTPUT_DIR / safe_filename(model, task)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(test_result, f, indent=2, ensure_ascii=False)

            gpu_stats = test_result.get("gpu_stats", {})
            all_summary.append({
                "Model": model,
                "Task": task,
                "Success": test_result.get("success", False),
                "TPS": test_result.get("tps", 0),
                "Eval_Tokens": test_result.get("eval_count", 0),
                "Prompt_Tokens": test_result.get("prompt_eval_count", 0),
                "Total_Sec": test_result.get("total_duration", 0),
                "Load_Sec": test_result.get("load_duration", 0),
                "Eval_Sec": test_result.get("eval_duration", 0),
                "Wall_Sec": test_result.get("wall_time", 0),
                "VRAM_Max_MB": gpu_stats.get("vram_max", 0),
                "VRAM_Avg_MB": gpu_stats.get("vram_avg", 0),
                "GPU_Util_Avg": gpu_stats.get("util_avg", 0),
                "GPU_Util_Peak": gpu_stats.get("util_peak", 0),
                "Temp_Peak_C": gpu_stats.get("temp_peak", 0),
                "Output_JSON": str(output_path),
                "Error": test_result.get("error", ""),
            })

            time.sleep(COOLDOWN_SEC)

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("批次測試完成！")
    print(f"總表已存至: {SUMMARY_CSV}")
    print(f"個別結果已存至: {OUTPUT_DIR}")
    print("=" * 60)
    print(summary_df)


if __name__ == "__main__":
    main()
