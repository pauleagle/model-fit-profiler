# Model Fit Profiler

Model Fit Profiler 是一個給本地 LLM / Ollama 使用的模型適性評測工具。它的目標不是只看模型誰最大或誰回答最好，而是把「任務類型、system prompt、回答品質、TPS、VRAM、GPU 使用率、溫度」放在同一套流程裡比較，最後產生可給 `llm-router` 使用的 routing recommendations。

## 核心概念

```text
Phase 1：跑模型 × task_type 測試
  ↓
Phase 2：LLM-as-a-Judge 評估回答品質
  ↓
Phase 3：整合品質分數與效能數據，產生 routing_recommendations.json
```

在這個專案中，`task_type` 不是單純分類標籤，而是以下設定的組合：

```text
task_type = system prompt + test_plan_task_question + temperature + max tokens + judge rubric + routing policy
```

其中 `task_system_prompts.json` 可與實際的 `llm-router` 共用；`profiler_task_prompts.json` 則只放測試計畫問題，欄位名稱為 `test_plan_task_question`，避免和 runtime prompt 混淆。

## 專案結構

```text
model-fit-profiler/
├─ phase1-profiler-batch.py          # Phase 1：模型批次測試與硬體監控
├─ phase2-llm-as-a-judge.py          # Phase 2：依 task_type 評分回答品質
├─ phase3-routing-recommender.py     # Phase 3：產生 llm-router 建議設定
├─ profiler_common.py                # 共用 helper：config include、分數正規化、JSON parser 等
├─ profiler_config.json              # 主設定：params、judge rubrics、weights、judge models、includes
├─ task_system_prompts.json          # task_type system prompts，可與 llm-router 共用
├─ profiler_task_prompts.json        # test_plan_task_question：Phase 1 測試題庫
├─ profiler_test_suite.json          # model × task_type 測試矩陣
├─ requirements.txt
└─ .gitignore
```


## 設定檔拆分

`profiler_config.json` 透過 `includes` 載入可共用或可替換的設定：

```json
{
  "includes": {
    "system_prompts": "task_system_prompts.json",
    "test_plan_task_question": "profiler_task_prompts.json",
    "test_suite": "profiler_test_suite.json"
  }
}
```

可用環境變數切換不同設定檔：

```bash
set PROFILER_SYSTEM_PROMPTS=task_system_prompts.local.json
set PROFILER_TEST_PLAN_TASK_QUESTION=profiler_task_prompts.quick.json
set PROFILER_TEST_SUITE=profiler_test_suite.quick.json
python phase1-profiler-batch.py
```

`PROFILER_TASK_PROMPTS` 仍保留為 `PROFILER_TEST_PLAN_TASK_QUESTION` 的 backward-compatible alias。

## 環境需求

- Python 3.10+
- Ollama
- 已下載要測試的模型
- Windows + NVIDIA GPU 時可透過 `nvidia-smi` 取得 GPU / VRAM / 溫度資料

安裝 Python 套件：

```bash
pip install -r requirements.txt
```

確認 Ollama 正常運作：

```bash
ollama list
ollama serve
```

## 快速開始

### 1. 設定測試矩陣

編輯 `profiler_test_suite.json`：

```json
{
  "test_suite": [
    {"model": "gemma3:1b", "tasks": ["router", "short_question"]},
    {"model": "phi3:mini", "tasks": ["analysis", "knowledge_refine"]},
    {"model": "deepseek-coder:6.7b-instruct", "tasks": ["coding", "debug"]}
  ]
}
```

### 2. 執行 Phase 1

```bash
python phase1-profiler-batch.py
```

預設輸出：

```text
./phase1_results/*.json
./phase1_results/phase1_summary.csv
```

Phase 1 會紀錄：

- 模型回答
- TPS
- prompt / eval token 數
- load duration / wall time
- GPU util
- VRAM max
- temperature peak

### 3. 執行 Phase 2

```bash
python phase2-llm-as-a-judge.py
```

預設輸出：

```text
./phase2_results/*.phase2_judge.json
./phase2_results/phase2_summary.csv
./phase2_results/phase2_summary.json
./phase2_results/phase2_leaderboard_by_task.json
```

Phase 2 特色：

- 依 task_type 使用不同 judge rubric
- router 使用專用評分邏輯，不會因為「沒有回答原始問題」被誤扣分
- 自動修正 judge 偶爾輸出 `0~1` 分制的問題
- 依 task_type weight 重新計算 `weighted_final_score`
- coding / debug 可使用 coding 專門 judge
- 可選擇 second pass judge 複評模糊案例

### 4. 執行 Phase 3

```bash
python phase3-routing-recommender.py
```

預設輸出：

```text
./phase3_results/routing_recommendations.json
./phase3_results/routing_candidates.csv
./phase3_results/routing_recommendations.md
```

Phase 3 會整合：

- Phase 1 效能資料
- Phase 2 品質分數
- task policy，例如 router / short_question 偏重速度與 VRAM，draft_generation 偏重品質

## 設定檔說明

### `profiler_config.json`

主設定檔，包含：

- `task_params`
- `score_keys`
- `task_weights`
- `task_rubrics`
- `judge_system_prompts`
- `judge_models`
- `routing_recommendations`
- `includes`

其中 `includes` 預設為：

```json
{
  "includes": {
    "system_prompts": "task_system_prompts.json",
    "test_plan_task_question": "profiler_task_prompts.json",
    "test_suite": "profiler_test_suite.json"
  }
}
```

### `task_system_prompts.json`

放各 task_type 的 runtime system prompt，可與 `llm-router` 共用。

### `profiler_task_prompts.json`

放 Phase 1 測試計畫題目，頂層欄位為 `test_plan_task_question`。適合替換成不同 benchmark 題庫。

### `profiler_test_suite.json`

放模型與 task_type 的測試矩陣。適合製作 quick / full / experimental 不同版本。

## 常用環境變數

### Phase 1

```bash
set PROFILER_CONFIG=profiler_config.json
set PROFILER_SYSTEM_PROMPTS=task_system_prompts.json
set PROFILER_TEST_PLAN_TASK_QUESTION=profiler_task_prompts.json
set PROFILER_TEST_SUITE=profiler_test_suite.json
set PHASE1_RESULTS_DIR=./phase1_results
set PHASE1_SUMMARY_CSV=./phase1_results/phase1_summary.csv
set OLLAMA_GENERATE_URL=http://localhost:11434/api/generate
```

### Phase 2

```bash
set PHASE1_RESULTS_DIR=./phase1_results
set PHASE2_RESULTS_DIR=./phase2_results
set PHASE2_DEFAULT_JUDGE=gemma3:4b
set PHASE2_CODING_JUDGE=deepseek-coder:6.7b-instruct
set PHASE2_SECOND_PASS_JUDGE=mistral:7b-instruct
set PHASE2_ENABLE_SECOND_PASS=true
```

PowerShell 寫法：

```powershell
$env:PHASE2_ENABLE_SECOND_PASS="false"
python phase2-llm-as-a-judge.py
```

## 目前預設任務類型

- `router`
- `short_question`
- `draft_generation`
- `analysis`
- `coding`
- `debug`
- `summarization`
- `knowledge_refine`
- `prompt_engineering`
- `general`

## v1.0.0 預期用途

這個版本適合用來建立第一版本地模型 routing baseline，例如：

```json
{
  "router": {"primary": "gemma3:1b", "fallback": "phi3:mini"},
  "short_question": {"primary": "gemma3:1b", "fallback": "llama3.2:3b"},
  "analysis": {"primary": "phi3:mini", "quality": "mistral:7b-instruct"},
  "coding": {"primary": "deepseek-coder:6.7b-instruct"}
}
```

實際結果請以你自己機器的 Phase 1 / 2 / 3 輸出為準。

## 注意事項

- LLM-as-a-Judge 不是絕對真理，建議用於相對比較與 routing baseline。
- 小模型 judge 可能會有評分漂移，所以 Phase 2 內建 score normalization 與 second pass。
- 不同硬體上的 TPS / VRAM 結果不可直接互相比較。
- `qwen3.6:27b` 這類大型模型可做高品質抽樣複評，但不建議在 4GB VRAM 裝置上作為日常 judge。
