# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
import torch
import pandas as pd
import logging
from typing import List, Optional
from metrics import MultiTurnInstructionFollowingPromptSolution
import json 
import numpy as np
from dataclasses import dataclass
# from vllm import LLM
# from vllm.sampling_params import SamplingParams

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass 
class GenerationSetting:
    max_new_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.9
    seed: int = 42





    

    


    
