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
import os

from openai import OpenAI, AsyncOpenAI

from utils import GenerationSetting


def get_api_bot(model_name, generation_setting, vllm_ip: str = "", async_bot: bool = False):
    if OpenAIBot.check_name(model_name):
        return OpenAIBot(model_name, generation_config=generation_setting)
    elif SwissAIBot.check_name(model_name):
        return SwissAIBot(model_name, generation_setting)
    elif VLLMAIBot.check_name(model_name):
        return VLLMAIBot(model_name, generation_setting, vllm_ip)
    else:
        raise NotImplementedError(f"The model {model_name} is not supported yet.")

class APIBot:

    def __init__(self, model, generation_config):
        self.model_name = model
        self.generation_config = generation_config

    def generate(self, messages):
        ...

    def check_name(self, name):
        ...


class OpenAIBot(APIBot):
    def __init__(self, model, generation_config):
        super().__init__(model, generation_config)
        self.client = OpenAI()

    def generate(self, messages) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=self.generation_config.max_new_tokens,
            seed=self.generation_config.seed,
            top_p=self.generation_config.top_p,
            temperature=self.generation_config.temperature
        )
        return response.choices[0].message.content

    @staticmethod
    def check_name(name):
        if name in ['o1-preview',
                    'o1-mini',
                    'o1-preview-2024-09-12',
                    'o1-mini-2024-09-12',
                    'gpt-4-turbo',
                    'gpt-4-turbo-2024-04-09',
                    'gpt-4-turbo-preview',
                    'gpt-4-0125-preview',
                    'gpt-4-1106-preview',
                    'gpt-4',
                    'gpt-4-0613',
                    'gpt-4o-2024-08-06'
                ]:
            return True
        return False

class SwissAIBot(APIBot):
    def __init__(self, model, generation_config):
        super().__init__(model, generation_config)
        api_key = os.environ.get("MY_SWISSAI_API")
        if not api_key:
            raise RuntimeError(f"Missing API key: MY_SWISSAI_API")
        self.client = OpenAI(api_key=api_key, base_url="https://api.swissai.cscs.ch/v1")

    def generate(self, messages) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=self.generation_config.max_new_tokens,
            seed=self.generation_config.seed,
            top_p=self.generation_config.top_p,
            temperature=self.generation_config.temperature
        )
        return response.choices[0].message.content

    @staticmethod
    def check_name(name):
        if name in ['swiss-ai/Apertus-8B-Instruct-2509',
                    'swiss-ai/Apertus-70B-Instruct-2509'
                ]:
            return True
        return False

class VLLMAIBot(APIBot):
    def __init__(self, model, generation_config, vllm_ip):
        super().__init__(model, generation_config)
        assert vllm_ip, f"local vllm ip should be set..."
        self.async_client = AsyncOpenAI(api_key="none", base_url=f"http://{vllm_ip}:8080/v1")   
        self.sync_client = OpenAI(api_key="none", base_url=f"http://{vllm_ip}:8080/v1")
        print(f"default completion kwargs: {self._chat_completion_kwargs(messages=[])}")

    def _chat_completion_kwargs(self, messages) -> dict:
        return dict(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=self.generation_config.max_new_tokens,
            seed=self.generation_config.seed,
            top_p=self.generation_config.top_p,
            temperature=self.generation_config.temperature,
            presence_penalty=self.generation_config.presence_penalty,
            extra_body={
                "top_k": self.generation_config.top_k,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    
    def generate(self, messages) -> str:
        response = self.sync_client.chat.completions.create(
            **self._chat_completion_kwargs(messages)
        )
        return response.choices[0].message.content

    async def async_generate(self, messages) -> str:
        response = await self.async_client.chat.completions.create(
            **self._chat_completion_kwargs(messages)
        )
        return response.choices[0].message.content

    @staticmethod
    def check_name(name):
        if name in [
            'Qwen3-8B-base', 
            'Qwen3-8B-sft-ifo', 
            'Qwen3-8B-sft-mix', 
            'Qwen3-8B-sft-mix-sum', 
            'Qwen3-8B-sft-mix-scaled',
            'Qwen3-8B-sft-mco-scaled',
            'Qwen3-8B-sft-mco-scaled-half',
            'Qwen3-8B-sft-mco-scaled-aggressive',
        ]:
            return True
        return False


if __name__ == '__main__':
    generation_setting = GenerationSetting(max_new_tokens=1024, temperature=0.6, top_p=0.9)
    bot = get_api_bot('Qwen3-8B-base', generation_setting, vllm_ip="172.28.37.8")
    history = [
        {'role': 'user', 'content': 'create an equation.'},
        {'role': 'assistant', 'content': 'x^2-4x+4=0'},
        {'role': 'user', 'content': 'solve the equation.'}
    ]
    print(bot.generate(history))
