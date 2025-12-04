# multi-challenge-adapt

快速上手
- 安装 Python 3.10+，依赖：`pip install -U openai datasets`.
- 配置 API：`export MY_SWISSAI_API=<your_key>`（默认基于 SwissAI `BASE_URL`，可在 `config.py` 修改）。
- 直接运行全流程（规划 + 生成对话 + 评估）：`python pipeline.py`
- 仅生成规划：`python generate_plans_only.py`
- 产物：`logs/` 保存日志，`generated/plans/` 与 `generated/dialogues/` 保存 jsonl 输出（按 run_id 命名）。

主要超参（见 `config.py`，可直接修改后运行）
- API/模型：`BASE_URL`，`API_KEY_ENV`，`PLANNER_MODEL`，`RESPONDER_MODEL`，`EVALUATOR_MODEL`。
- 规模控制：`SEEDS_PER_RUN`，`PLANS_PER_SEED`，`TRAJECTORIES_PER_PLAN`，`PLAN_TURNS`。
- 生成温度/长度：`PLANNER_TEMPERATURE`，`RESPONDER_TEMPERATURE`，`EVALUATOR_TEMPERATURE`，`PLANNER_MAX_TOKENS`，`RESPONDER_MAX_TOKENS`，`PLANNER_PROMPT_MAX_CHARS`，`RESPONDER_PROMPT_MAX_CHARS`，`MAX_TOKENS`，`MAX_RETRIES`，`DEFAULT_TIMEOUT`。
- 并发：`MAX_CONCURRENCY`（限制同时请求数）。
- 提示增强开关：`USE_TAXONOMY`，`USE_FEWSHOT`，`USE_AGENT_TEMPLATE`，以及 `FEWSHOT_SAMPLES`，`FEWSHOT_MAX_CHARS`，`TAXONOMY_SAMPLE_RATIO`。

自包含与数据
- Hugging Face 数据集：`allenai/tulu-3-sft-personas-instruction-following`（seed）与 `nmayorga7/multichallenge`（few-shot，默认开启）会自动下载到 `hfdata/`；若离线，可预先把对应缓存目录拷贝到同名路径。
- 附录提示文件（taxonomy、planner template）已复制到仓库内的 `appendix_items/`，`config.py` 默认指向此处，无需依赖仓库外的目录。
