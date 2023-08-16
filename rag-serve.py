#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import sys
import logging
from aiohttp import web
from pybars import Compiler
import openai

openai.api_key = "sk-***"  # your openai api key you registered on the openai.

from langchain.embeddings import HuggingFaceEmbeddings

from llama_index import StorageContext, ServiceContext, load_index_from_storage
from llama_index.embeddings import LangchainEmbedding
from llama_index import set_global_service_context
from llama_index.response.schema import RESPONSE_TYPE

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

routes = web.RouteTableDef()

@routes.get("/")
async def hello(_: web.Request):
    return web.Response(text="Hello, world")


def conversation(prompt, turns):
    res = [{
        "role": "system",
        "content": prompt
    }]
    res.extend(turns)
    return res


# curl -v -d 'input=中国有多大' http://127.0.0.1:8080/query
@routes.post("/query")
async def query(request: web.Request):
    req = await request.post()
    prompt = req.get("prompt")
    turns = req.get("turns")
    size = len(turns)
    if len(turns) == 0:
        return web.json_response({"errMsg": f'input type is not str'})
    if turns[size - 1].role != "user":
        return web.json_response({"errMsg": f'last turn is not from user'})

    user_input = turns[size - 1].text

    retriever = request.app['engine']

    context = retriever.query(user_input)
    compiler = request.app['compiler'].compiler
    template = compiler.compile(prompt)
    new_prompt = template({
        "query": query,
        "context": context})

    response = openai.ChatCompletion.create(
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
