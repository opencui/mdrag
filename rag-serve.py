#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import openai

from aiohttp import web, ClientSession
from pybars import Compiler
from llama_index import set_global_service_context
from llama_index.embeddings import LangchainEmbedding
from llama_index import StorageContext, ServiceContext, load_index_from_storage
from langchain.embeddings import HuggingFaceEmbeddings

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

routes = web.RouteTableDef()


@routes.get("/")
async def hello(_: web.Request):
    return web.Response(text="Hello, world")


def conversation(prompt, turns):
    res = [{"role": "system", "content": prompt}]
    res.extend(turns)
    return res


# curl -v -d 'input=中国有多大' http://127.0.0.1:8080/query
@routes.post("/query")
async def query(request: web.Request):
    req = await request.json()
    turns = req.get("turns", [])
    prompt = req.get("prompt", "")
    if len(prompt) == 0:
        prompt = request.app['prompt']
    if len(turns) == 0:
        return web.json_response({"errMsg": f'input type is not str'})
    if turns[-1].get("role", "") != "user":
        return web.json_response({"errMsg": f'last turn is not from user'})

    user_input = turns[-1].get("content", "")

    retriever = request.app['engine']

    context = retriever.retrieve(user_input)
    compiler = request.app['compiler']
    template = compiler.compile(prompt)
    new_prompt = template({"query": user_input, "context": context})

    async with ClientSession(trust_env=True) as session:
        openai.aiosession.set(session)
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=conversation(new_prompt, turns),
            temperature=0  # Try to as deterministic as possible.
        )

    resp = {"reply": response.choices[0].message["content"]}
    return web.json_response(resp)


def init_app(index):
    app = web.Application()
    app.add_routes(routes)
    app['engine'] = index.as_retriever()
    app['compiler'] = Compiler()
    app['prompt'] = "We have provided context information below. \n" \
        "---------------------\n"\
        "{{context}}"\
        "\n---------------------\n"\
        "Given this information, please answer the question: {{query}}\n"
    return app


# OPENAI_API_KEY=xxx python index_path
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    p = sys.argv[1]

    if not os.path.isdir(p):
        sys.exit(1)

    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    langchain_embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
    )

    embed_model = LangchainEmbedding(langchain_embedding)
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    set_global_service_context(service_context)

    storage_context = StorageContext.from_defaults(persist_dir=p)
    index = load_index_from_storage(storage_context)
    web.run_app(init_app(index))
