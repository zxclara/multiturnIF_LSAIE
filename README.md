# multi-challenge-adapt

Quick start
- Python 3.10+, install deps: `pip install -U openai datasets`.
- Configure endpoints in `config.py` (`PLANNER_BASE_URL`, `RESPONDER_BASE_URL`, `EVALUATOR_BASE_URL`) and adjust `SEEDS_PER_RUN` as needed.
- Full pipeline (plans + dialogues + evaluation): `python async_pipeline.py`
- Plans only: `python generate_plans_only.py`
- Outputs: logs in `logs/`, plans in `generated/plans/`, dialogues in `generated/dialogues/` (each file named by `run_id`).

What the pipeline does (async_pipeline)
- Loads seeds from `allenai/tulu-3-sft-personas-instruction-following`.
- For each seed, generates `PLANS_PER_SEED` plans, rotating challenge categories in order: Reliable Version Editing → Instruction Retention → Inference Memory → Self-Coherence. Each plan injects the corresponding appendix category details.
- Few-shot examples are filtered by the plan’s category axis (from `nmayorga7/multichallenge`), respecting `FEWSHOT_SAMPLES` and `FEWSHOT_MAX_CHARS`.
- Each plan drives `TRAJECTORIES_PER_PLAN` responder dialogues; trajectories are evaluated and scored, with top non-triggered traps marked `selected`.
- All plans/trajectories are written; no truncation by zip.

Key knobs (edit `config.py`)
- API/models: `PLANNER_MODEL`, `RESPONDER_MODEL`, `EVALUATOR_MODEL`, endpoint URLs, `API_KEY_ENV`.
- Scale: `SEEDS_PER_RUN`, `PLANS_PER_SEED`, `TRAJECTORIES_PER_PLAN`, `PLAN_TURNS`.
- Generation: `PLANNER_TEMPERATURE`, `RESPONDER_TEMPERATURE`, `EVALUATOR_TEMPERATURE`, token/char limits per role, `MAX_RETRIES`, `DEFAULT_TIMEOUT`.
- Concurrency: `MAX_CONCURRENCY`.
- Prompt extras: `USE_FEWSHOT`, `FEWSHOT_SAMPLES`, `FEWSHOT_MAX_CHARS`, `USE_AGENT_TEMPLATE` (agent template optional). Legacy toggles for taxonomy sampling are commented out.

Data & resources
- Hugging Face datasets cache to `hfdata/` (seed + few-shot). If offline, copy the cached dirs into the same path.
- Appendix prompt files live in `appendix_items/` (taxonomy, planner template); `config.py` already points to them.
