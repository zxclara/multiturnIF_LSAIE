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

from openai import OpenAI

from utils import GenerationSetting


def get_api_bot(model_name, generation_setting):
    if OpenAIBot.check_name(model_name):
        return OpenAIBot(model_name, generation_config=generation_setting)
    elif SwissAIBot.check_name(model_name):
        return SwissAIBot(model_name, generation_setting)
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


if __name__ == '__main__':
    generation_setting = GenerationSetting(max_new_tokens=1024, temperature=0.6, top_p=0.9)
    bot = get_api_bot('swiss-ai/Apertus-8B-Instruct-2509',generation_setting)
    history = [
        {'role': 'user', 'content': 'create an equation.'},
        {'role': 'assistant', 'content': 'x^2-4x+4=0'},
        {'role': 'user', 'content': 'solve the equation.'}
    ]
    print(bot.generate(history))
