#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass

from typing import Any, List, Optional
import logging
import gin
from openai import AsyncOpenAI


@gin.configurable
def get_generator(model, openai_base_url=None, openai_api_key=None):
    logging.info(f"Using {model}:{openai_base_url} as generator...")
    models = model.split("/")
    if models[0] == "openai":
        return OpenAIGenerator(
            model=models[1],
            url=openai_base_url,
            api_key=openai_api_key,
        )
    elif model.startswith(".") and model.endswith("bin"):
        return LlamaGenerator(model)
    else:
        return HuggingFaceGenerator(model)


@dataclass
class Response:
    reply: str


class OpenAIGenerator:
    def __init__(self, model="gpt-3.5-turbo", temperature=0, url=None, api_key=None):
        logging.info(f"OpenAIGenerator: {model}:{url}")
        self.url = url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    @classmethod
    def conversation(cls, prompt, turns):
        res = [{"role": "system", "content": prompt}]
        res.extend(turns)
        return res

    async def agenerate(self, system_prompt, turns) -> Response:
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.url)
        response = await client.chat.completions.create(
            temperature=0,  # Try to as deterministic as possible.
            model=self.model,
            messages=OpenAIGenerator.conversation(system_prompt, turns),
        )

        return Response(response.choices[0].message.content)


def llama2_prompt(system_prompt, turns):
    res = f"""<s>[INST] <<SYS>>{system_prompt}<</SYS>>\n"""
    res += f"""{turns[0]["content"]}  [/INST] """
    num_turns = int(len(turns) / 2)
    for i in range(num_turns):
        res += f"""{turns[2*i + 1]["content"]} </s><s>[INST] {turns[2*i + 2]["content"]} [/INST]"""
    return res


class LlamaGenerator:
    def __init__(self, model_path, n_ctx=4096):
        from llama_cpp import Llama

        self.llm = Llama(model_path=model_path, n_ctx=n_ctx)
        self.max_tokens = 512
        self.temperature = 0
        self.top_p = 0.5
        self.echo = False
        self.stop = ["#"]

    async def agenerate(self, system_prompt, turns) -> Response:
        prompt = llama2_prompt(system_prompt, turns)
        output = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            echo=self.echo,
            stop=self.stop,
            temperature=self.temperature,
        )
        return Response(output["choices"][0]["text"])


@gin.configurable
class HuggingFaceGenerator:
    """HuggingFace generator."""

    def __init__(
        self,
        model_name: str,
        context_window: int = 32 * 1024,
        max_new_tokens: int = 256,
        device_map: str = "auto",
        stopping_ids: Optional[List[int]] = None,
        tokenizer_kwargs: Optional[dict] = None,
        tokenizer_outputs_to_remove: Optional[list] = None,
        model_kwargs: Optional[dict] = None,
        generate_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize params."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            StoppingCriteria,
            StoppingCriteriaList,
        )

        model_kwargs = model_kwargs or {}
        self._model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **model_kwargs,
        )

        # check context_window
        config_dict = self.model.config.to_dict()
        model_context_window = int(
            config_dict.get("max_position_embeddings", context_window)
        )
        if model_context_window and model_context_window < context_window:
            logging.warning(
                f"Supplied context_window {context_window} is greater "
                "than the model's max input size {model_context_window}. "
                "Disable this warning by setting a lower context_window."
            )
            context_window = model_context_window

        tokenizer_kwargs = tokenizer_kwargs or {}
        if "max_length" not in tokenizer_kwargs:
            tokenizer_kwargs["max_length"] = context_window

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

        self._context_window = context_window
        self._max_new_tokens = max_new_tokens

        self._generate_kwargs = generate_kwargs or {}
        self._device_map = device_map
        self._tokenizer_outputs_to_remove = tokenizer_outputs_to_remove or []
        self._total_tokens_used = 0
        self._last_token_usage: Optional[int] = None

        # setup stopping criteria
        stopping_ids_list = stopping_ids or []

        class StopOnTokens(StoppingCriteria):
            def __call__(
                self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
                **kwargs: Any,
            ) -> bool:
                for stop_id in stopping_ids_list:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        self._stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    async def agenerate(self, system_prompt, turns) -> Response:
        """Completion endpoint."""
        full_prompt = llama2_prompt(system_prompt, turns)
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        # remove keys from the tokenizer if needed, to avoid HF errors
        for key in self._tokenizer_outputs_to_remove:
            if key in inputs:
                inputs.pop(key, None)

        tokens = self.model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            stopping_criteria=self._stopping_criteria,
            **self._generate_kwargs,
        )
        completion_tokens = tokens[0][inputs["input_ids"].size(1) :]
        self._total_tokens_used += len(completion_tokens) + inputs["input_ids"].size(1)
        completion = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)

        return Response(completion)
