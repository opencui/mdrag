#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dataclasses
import logging
import os
import os.path
import tarfile
import io
import shutil
import sys
import tempfile
import pickle
import base64
import time

import gin
from lru import LRU
from aiohttp import web
from contextlib import closing
from llama_index.core import (
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    set_global_service_context,
)
from pybars import Compiler
from torch import mode

from rag_index import build_index
from processors.embedding import get_embedding
from processors.llm import get_generator
from processors.retriever import HybridRetriever

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

routes = web.RouteTableDef()


@routes.get("/")
async def hello(_: web.Request):
    return web.Response(text="Hello, world")


def get_agent_path(req: web.Request) -> str:
    data_path = req.app["data_path"]
    org_name = req.match_info["org"]
    agent_name = req.match_info["agent"]
    agent_path = os.path.join(data_path, org_name, agent_name)
    return agent_path


@gin.configurable
def get_retriever(req: web.Request, mode: str):
    agent_path = get_agent_path(req)
    lru_cache = req.app["lru_cache"]

    if agent_path in lru_cache and lru_cache[agent_path] is not None:
        return lru_cache[agent_path]

    match mode:
        case "hybrid":
            storage_context = StorageContext.from_defaults(persist_dir=agent_path)
            keyword = load_index_from_storage(
                storage_context, index_id="keyword"
            ).as_retriever()
            embedding = load_index_from_storage(
                storage_context, index_id="embedding"
            ).as_retriever()
            retriever = HybridRetriever(embedding, keyword)  # type: ignore
            lru_cache[agent_path] = retriever

            return retriever
        case "embedding":
            storage_context = StorageContext.from_defaults(persist_dir=agent_path)
            retriever = load_index_from_storage(
                storage_context, index_id="embedding"
            ).as_retriever()
            lru_cache[agent_path] = retriever

            return retriever
        case "keyword":
            storage_context = StorageContext.from_defaults(persist_dir=agent_path)
            retriever = load_index_from_storage(
                storage_context, index_id="keyword"
            ).as_retriever()
            lru_cache[agent_path] = retriever

            return retriever
        case _:
            return None


@routes.get("/index/{org}/{agent}")
async def check(request: web.Request):
    agent_path = get_agent_path(request)
    if not os.path.exists(agent_path):
        return web.json_response({"errMsg": "index not found"}, status=500)
    return web.json_response({})


@routes.post("/index/{org}/{agent}")
async def build_index_handler(request: web.Request):
    agent_path = get_agent_path(request)
    lru_cache = request.app["lru_cache"]

    if os.path.exists(agent_path):
        shutil.rmtree(agent_path)
    os.makedirs(agent_path)

    with tempfile.TemporaryDirectory(prefix="mdrag_") as tmpdirname:
        args = []

        reader = await request.multipart()
        o = await reader.next()
        while o is not None:
            match o.name:  # type: ignore
                case "url":
                    args.append(await o.text())  # type: ignore
                case "tar" | "tar.gz":
                    n = tempfile.mkdtemp(dir=tmpdirname)
                    b = io.BytesIO(await o.read())  # type: ignore
                    with closing(tarfile.open(fileobj=b)) as t:
                        t.extractall(n)
                    args.append(n)
                case "file":
                    _p = os.path.abspath(os.path.join(tmpdirname, o.filename))
                    if not _p.startswith(tmpdirname):
                        return web.json_response(
                            {"errMsg": f"file name error {o.filename}"}
                        )

                    os.makedirs(os.path.dirname(_p), exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        dir=os.path.dirname(_p),
                        suffix=f"_{os.path.basename(_p)}",  # type: ignore
                        delete=False,
                    ) as f:
                        f.write(await o.read())  # type: ignore
                        args.append(f.name)

            o = await reader.next()

        print(args)
        embedding_model = request.app["embedding_model"]

        headers = {}
        for k in request.headers.items():
            if len(k) >= 2:
                headers[k[0]] = k[1]
        with open(os.path.join(agent_path, "headers.pickle"), "wb") as f:
            pickle.dump(headers, f)

        build_index(embedding_model, agent_path, *args)
        lru_cache[agent_path] = None

    return web.json_response({})


@routes.post("/v1/tryitnow/")
async def tryitnow(request: web.Request):
    req = await request.json()

    text = req.get("text", "")
    events = req.get("events", [])
    initial = req.get("initial", "")

    if len(text) == 0:
        return web.json_response({"errMsg": "text value is not empty"})

    if not isinstance(initial, bool):
        return web.json_response({"errMsg": "initial type is not bool"})

    if not isinstance(events, list):
        return web.json_response({"errMsg": "events type is not list"})

    for e in events:
        if not isinstance(e, dict):
            return web.json_response({"errMsg": "events[index] type is not dict"})

    return web.json_response({})


@routes.post("/query/{org}/{agent}")
async def query(request: web.Request):
    start_time = time.time()
    agent_path = get_agent_path(request)

    with open(os.path.join(agent_path, "headers.pickle"), "rb") as f:
        headers = pickle.load(f)

    knowledge_key = headers.get("Knowledge-Key")
    knowledge_url = headers.get("Knowledge-Url")
    knowledge_model = headers.get("Knowledge-Model").lower()
    knowledge_model_name = headers.get("Knowledge-Model-Name")
    knowledge_mode_prompt = headers.get("Knowledge-Model-Prompt")

    logging.info("headers")
    logging.info(headers)

    if knowledge_model == "": knowledge_model = "openai"

    if knowledge_model_name is None:
        logging.info("could not find the model name")
        return web.json_response({"errMsg": "model name found"})

    knowledge_model_name = f"{knowledge_model}/{knowledge_model_name}"

    try:
        if knowledge_mode_prompt is not None:
            knowledge_mode_prompt = base64.b64decode(knowledge_mode_prompt).decode()
    except Exception as e:
        logging.error(e)

    if not os.path.exists(agent_path):
        return web.json_response({"errMsg": "index not found"})

    req = await request.json()
    logging.info("request")
    logging.info(req)

    turns = req.get("turns", [])
    prompt = req.get("prompt", "")

    if len(prompt) == 0:
        prompt = request.app["prompt"]

    if knowledge_mode_prompt is not None:
        prompt = knowledge_mode_prompt

    if not isinstance(turns, list):
        logging.error("turns is not a list")
        return web.json_response({"errMsg": "turns type is not list"})

    if len(turns) == 0:
        logging.error("empty turns")
        feedback = req.get("feedback", None)
        if feedback:
            return web.json_response({"reply": ""})
        return web.json_response({"errMsg": "turns length cannot be empty"})

    if turns[-1].get("role", "") != "user":
        logging.info("last turn is not from user")
        return web.json_response({"errMsg": "last turn is not from user"})

    user_input = turns[-1].get("content", "")

    retriever = get_retriever(request)  # type: ignore
    # What is the result here?
    context = retriever.retrieve(user_input)

    template = request.app["compiler"].compile(prompt)

    new_prompt = template({"query": user_input, "context": context})
    logging.info("new_prompt")
    logging.info(new_prompt)
    logging.info(f"knowledge_model:{knowledge_model}")
    logging.info(f"model_name: {knowledge_model_name}")
    logging.info(f"llm_url: {knowledge_url}")
    match knowledge_model:
        case "openai":
            llm = get_generator(  # type: ignore
                model=knowledge_model_name,
                openai_api_key=knowledge_key,
                openai_base_url=knowledge_url,
            )
        case _:
            return web.json_response(
                {"errMsg": f"knowledge model error: {knowledge_model}"}
            )

    # So that we can use different llm.
    resp = await llm.agenerate(new_prompt, turns)
    logging.info("resp")
    logging.info(resp)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time}")
    return web.json_response(dataclasses.asdict(resp))


@routes.post("/retrieve/{org}/{agent}")
async def retrieve(request: web.Request):
    req = await request.json()
    turns = req.get("turns", [])
    prompt = req.get("prompt", "")
    if len(prompt) == 0:
        prompt = request.app["prompt"]
    if len(turns) == 0:
        return web.json_response({"errMsg": "input type is not str"})
    if turns[-1].get("role", "") != "user":
        return web.json_response({"errMsg": "last turn is not from user"})

    user_input = turns[-1].get("content", "")

    retriever = get_retriever(request)  # type: ignore

    # What is the result here?
    context = retriever.retrieve(user_input)

    resp = {"reply": context}
    return web.json_response(resp)


def init_app(data_path, embedding_model):
    app = web.Application()
    app.add_routes(routes)

    app["data_path"] = data_path
    app["lru_cache"] = LRU(512)

    app["embedding_model"] = embedding_model

    app["compiler"] = Compiler()
    app["prompt"] = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{{context}}"
        "\n---------------------\n"
        "Given this information, please answer the question: {{query}}\n"
    )
    return app


if __name__ == "__main__":
    gin.parse_config_file("serve.gin")
    embedding_model = get_embedding()  # type: ignore

    if len(sys.argv) != 2:
        sys.exit(0)

    p = sys.argv[1]
    if not os.path.isdir(p):
        sys.exit(1)

    service_context = ServiceContext.from_defaults(
        llm=None,
        llm_predictor=None,
        embed_model=embedding_model,
    )
    set_global_service_context(service_context)

    web.run_app(init_app(p, embedding_model))
