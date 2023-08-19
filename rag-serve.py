#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import openai

from aiohttp import web, ClientSession
from pybars import Compiler
from llama_index import set_global_service_context
from llama_index import StorageContext, ServiceContext, load_index_from_storage
from processors.embedding import get_embedding
from llama_index.indices.postprocessor import AutoPrevNextNodePostprocessor

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

    # What is the result here?
    context = retriever.retrieve(user_input)

    template = request.app['compiler'].compile(prompt)

    new_prompt = template({"query": user_input, "context": context})

    # We should consider using https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML
    # which can be used on the cpu as well.
    async with ClientSession(trust_env=True) as session:
        openai.aiosession.set(session)
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=conversation(new_prompt, turns),
            temperature=0  # Try to as deterministic as possible.
        )

    resp = {"reply": response.choices[0].message["content"]}
    return web.json_response(resp)


@routes.post("/retrieve")
async def retrieve(request: web.Request):
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

    # What is the result here?
    context = retriever.retrieve(user_input)

    resp = {"reply": context}
    return web.json_response(resp)


def init_app(embedding_index, keyword_index):
    app = web.Application()
    app.add_routes(routes)
    app['engine'] = embedding_index.as_retriever()
    app['compiler'] = Compiler()
    app['prompt'] = "We have provided context information below. \n" \
        "---------------------\n"\
        "{{context}}"\
        "\n---------------------\n"\
        "Given this information, please answer the question: {{query}}\n"
    return app


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Where is the index saved?")
        sys.exit(1)

    p = sys.argv[1]

    if not os.path.isdir(p):
        sys.exit(1)

    service_context = ServiceContext.from_defaults(
        embed_model=get_embedding())

    set_global_service_context(service_context)

    storage_context = StorageContext.from_defaults(persist_dir=p)
    embedding_index = load_index_from_storage(storage_context, index_id=0)
    keyword_index = load_index_from_storage(storage_context, index_id=1)
    web.run_app(init_app(index))
