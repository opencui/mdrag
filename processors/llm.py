#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from aiohttp import ClientSession
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
import openai


def get_generator():
    return OpenAIGenerator()


class OpenAIGenerator:
    def __init__(self, model="gpt-3.5-turbo", temperature=0):
        self.mmodel = model
        self.temperature = temperature

    @classmethod
    def conversation(cls, prompt, turns):
        res = [{"role": "system", "content": prompt}]
        res.extend(turns)
        return res

    async def agenerate(self, system_prompt, turns):
        async with ClientSession(trust_env=True) as session:
            openai.aiosession.set(session)
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=OpenAIGenerator.conversation(system_prompt, turns),
                temperature=0  # Try to as deterministic as possible.
            )
        return {"reply": response.choices[0].message["content"]}


class LocalGenerator:
    def __init__(self, context=4096):
        self.context = context

    async def agenerate(self, system_prompt, turns):
        return {"reply": "nothing"}
