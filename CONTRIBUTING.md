# Contributing

感謝你想協助改善 Model Fit Profiler。這個專案的核心目標是：用可重複、可比較、可調整的方式，評估本地 LLM 在不同 task_type 下的效能與回答品質。

## 開發原則

### 1. 設定優先外部化

請優先把可調整的內容放在 JSON 設定檔，而不是硬寫在 Python 裡。

適合放在 JSON 的內容：

- system prompt
- task prompt
- test suite
- task params
- judge rubric
- task weights
- routing recommendation baseline

適合放在 Python 的內容：

- 檔案讀寫流程
- Ollama / OpenAI-compatible API 呼叫
- GPU monitoring
- score normalization
- leaderboard / report generation

### 2. Phase 邊界清楚

請維持三個 phase 的職責分離：

```text
Phase 1：產生模型回答與效能資料
Phase 2：評估回答品質
Phase 3：根據品質與效能產生 routing recommendation
```

不要讓 Phase 1 直接做品質判斷，也不要讓 Phase 2 改寫 Phase 1 的原始結果。

### 3. task_type 必須 task-aware

新增 task_type 時，請同步補齊：

- `system_prompts`
- `task_params`
- `profiler_task_prompts.json`
- `task_rubrics`
- `task_weights`

如果沒有 rubric，Phase 2 的評分會退回 general，結果可能失真。

### 4. 保留原始資料

Phase 1 / Phase 2 輸出應保留足夠 metadata，方便後續追蹤：

- model
- task
- prompt
- response
- params
- judge model
- raw judge output
- weighted score
- TPS / VRAM / wall time

### 5. 不要把本機結果當絕對排名

TPS / VRAM / load time 會受到硬體、driver、Ollama 版本、模型量化格式影響。文件中請避免宣稱某模型絕對比較好，只能說「在目前測試環境與題目下」。

## 建議開發流程

1. 建立 branch。
2. 修改程式或 JSON 設定。
3. 先跑 quick test suite。
4. 檢查 Phase 1 / 2 / 3 是否都能產出結果。
5. 執行 Python 語法檢查：

```bash
python -m py_compile profiler_common.py phase1-profiler-batch.py phase2-llm-as-a-judge.py phase3-routing-recommender.py
```

6. 若修改輸出格式，請同步更新 README / CHANGELOG。

## 新增模型測試

請優先修改 `profiler_test_suite.json`：

```json
{
  "test_suite": [
    {"model": "your-model", "tasks": ["router", "analysis"]}
  ]
}
```

若只是本機實驗，不建議直接覆蓋主測試矩陣，可建立：

```text
profiler_test_suite.local.json
profiler_test_suite.quick.json
profiler_test_suite.full.json
```

並用環境變數指定：

```bash
set PROFILER_TEST_SUITE=profiler_test_suite.quick.json
python phase1-profiler-batch.py
```

## 新增 task_type

新增 task_type 時，請至少修改：

```text
profiler_config.json
profiler_task_prompts.json
```

必要項目：

```json
{
  "system_prompts": {},
  "task_params": {},
  "task_weights": {},
  "task_rubrics": {}
}
```

並在 `profiler_task_prompts.json` 補一個可重複測試的代表題目。

## Pull Request 建議格式

```text
## Summary
- 做了什麼修改

## Why
- 為什麼需要這個修改

## Test
- 執行了哪些 phase
- 是否跑過 py_compile

## Output Impact
- 是否改變 phase1 / phase2 / phase3 輸出格式
```

## Commit Message 建議

```text
feat: add phase3 routing recommender
fix: normalize judge score scale
refactor: split task prompts and test suite config
chore: rename phase1 script
```
