# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-05-07

### Added

- Added Phase 1 batch runner: `phase1-profiler-batch.py`.
- Added Phase 2 task-aware LLM-as-a-Judge runner: `phase2-llm-as-a-judge.py`.
- Added Phase 3 routing recommender: `phase3-routing-recommender.py`.
- Added shared helper module: `profiler_common.py`.
- Added externalized config files:
  - `profiler_config.json`
  - `profiler_task_prompts.json`
  - `profiler_test_suite.json`
- Added task-aware system prompts for:
  - `router`
  - `short_question`
  - `draft_generation`
  - `analysis`
  - `coding`
  - `debug`
  - `summarization`
  - `knowledge_refine`
  - `prompt_engineering`
- Added task-specific judge rubrics and score weights.
- Added score normalization for local judges that accidentally return `0~1` instead of `0~10`.
- Added weighted final score recomputation by task type.
- Added optional second-pass judge flow for ambiguous or low-confidence grading results.
- Added Phase 3 output files:
  - `phase3_results/routing_recommendations.json`
  - `phase3_results/routing_candidates.csv`
  - `phase3_results/routing_recommendations.md`
- Added GitHub-ready project docs:
  - `README.md`
  - `CHANGELOG.md`
  - `CONTRIBUTING.md`

### Changed

- Renamed Phase 1 script to hyphen naming: `phase1-profiler-batch.py`.
- Kept Phase 2 script naming as: `phase2-llm-as-a-judge.py`.
- Moved Phase 1 summary output into `./phase1_results/phase1_summary.csv` by default.
- Split `task_prompts` and `test_suite` into independent JSON files for reuse.
- Updated config loading to support include files and environment variable overrides.

### Notes

- This is the first GitHub-ready baseline release.
- Tested for Python syntax with `py_compile`.
- Designed primarily for local Ollama model comparison on constrained hardware.
