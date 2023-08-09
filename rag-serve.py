#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
from aiohttp import web

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


# curl -v -d 'input=中国有多大' http://127.0.0.1:8080/query
@routes.post("/query")
async def query(request: web.Request):
    req = await request.post()
    input = req.get("input")

    if type(input) != str:
        return web.json_response({"errMsg": f'input type is not str'})

    query_engine = request.app['engine']
    response: RESPONSE_TYPE = query_engine.query(input)

    resp = {"result": str(response)}
    return web.json_response(resp)


def init_app(index):
    app = web.Application()
    app.add_routes(routes)
    app['engine'] = index.as_query_engine()
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
