from pathlib import Path

# API configuration
PLANNER_BASE_URL = "http://172.28.36.128:8080/v1"
RESPONDER_BASE_URL = "http://172.28.44.24:8080/v1"
EVALUATOR_BASE_URL = "http://172.28.36.128:8080/v1"
API_KEY_ENV = "none"

# Models
PLANNER_MODEL = "openai/gpt-oss-120b" # w/o prefix caching
RESPONDER_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct" # w/ prefix caching
EVALUATOR_MODEL = "openai/gpt-oss-120b" # w/o prefix caching

# Generation knobs
PLANS_PER_SEED = 4
TRAJECTORIES_PER_PLAN = 4
PLAN_TURNS = 8
SEEDS_PER_RUN = 128

PLANNER_TEMPERATURE = 0.9
RESPONDER_TEMPERATURE = 0.7
EVALUATOR_TEMPERATURE = 0.3
MAX_RETRIES = 4
PLANNER_MAX_TOKENS = 4096
PLANNER_PROMPT_MAX_CHARS = 24000
FEWSHOT_MAX_CHARS = 8000

# Paths
ROOT = Path(__file__).resolve().parent
HF_CACHE = ROOT / "hfdata"
PLANS_DIR = ROOT / "generated" / "plans"
DIALOGUES_DIR = ROOT / "generated" / "dialogues"
LOG_DIR = ROOT / "logs"
TAXONOMY_FILES = [
    ROOT / "appendix_items" / "topic_heirarchical_tax.tex",
    ROOT / "appendix_items" / "evaluation_configs.tex",
]
PLANNER_TEMPLATE_FILE = ROOT / "appendix_items" / "agent_sys_prompts.tex"

# Concurrency settings (limit simultaneous API calls)
MAX_CONCURRENCY = 128

# Misc
DEFAULT_TIMEOUT = 300
MAX_TOKENS = 32768
RESPONDER_MAX_TOKENS = 2048
RESPONDER_PROMPT_MAX_CHARS = 12000
DATASET_ID = "allenai/tulu-3-sft-personas-instruction-following"
FEWSHOT_DATASET_ID = "nmayorga7/multichallenge"
FEWSHOT_SAMPLES = 2

# Prompt size toggles
USE_TAXONOMY = False
TAXONOMY_SAMPLE_RATIO = 0.1  # 1.0 = full text; e.g., 0.1 to sample ~10% segments
USE_FEWSHOT = True
USE_AGENT_TEMPLATE = False
