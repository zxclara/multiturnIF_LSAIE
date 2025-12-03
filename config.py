from pathlib import Path

# API configuration
BASE_URL = "https://api.swissai.cscs.ch/v1"
API_KEY_ENV = "MY_SWISSAI_API"

# Models (currently all set to Qwen3-32B, keep names swappable)
PLANNER_MODEL = "Qwen/Qwen3-32B"
RESPONDER_MODEL = "Qwen/Qwen3-32B"
EVALUATOR_MODEL = "Qwen/Qwen3-32B"

# Generation knobs
PLANS_PER_SEED = 4
TRAJECTORIES_PER_PLAN = 4
PLAN_TURNS = 15
SEEDS_PER_RUN = 10

PLANNER_TEMPERATURE = 0.9
RESPONDER_TEMPERATURE = 0.7
EVALUATOR_TEMPERATURE = 0.3
MAX_RETRIES = 4

# Paths
ROOT = Path(__file__).resolve().parent
HF_CACHE = ROOT / "hfdata"
PLANS_DIR = ROOT / "generated" / "plans"
DIALOGUES_DIR = ROOT / "generated" / "dialogues"
LOG_DIR = ROOT / "logs"
TAXONOMY_FILES = [
    ROOT.parent / "multichallenge_arXiv-2501.17399v2" / "appendix_items" / "topic_heirarchical_tax.tex",
    ROOT.parent / "multichallenge_arXiv-2501.17399v2" / "appendix_items" / "evaluation_configs.tex",
]

# Concurrency settings (limit simultaneous API calls)
# You measured success at 256 concurrent requests; set a conservative default here.
MAX_CONCURRENCY = 16

# Misc
DEFAULT_TIMEOUT = 300
MAX_TOKENS = 2048
DATASET_ID = "allenai/tulu-3-sft-personas-instruction-following"
FEWSHOT_DATASET_ID = "nmayorga7/multichallenge"
FEWSHOT_SAMPLES = 1

# Prompt size toggles
USE_TAXONOMY = False
USE_FEWSHOT = True
