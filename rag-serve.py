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
import time
from typing import Optional, Literal, Annotated, Union

import gin
from lru import LRU

from contextlib import closing
from llama_index.core import (
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    set_global_service_context,
)

from typing import Any, List
from jinja2 import Environment, Template
from pydantic import BaseModel, Field, TypeAdapter

from rag_index import build_index
from processors.embedding import get_embedding
from processors.llm import get_generator
from processors.retriever import HybridRetriever


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

routes = FastAPI()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class OpenAIMessage(BaseModel):
    role: str
    content: str


# All system1 should be one of these.
class KnowledgeTag(BaseModel):
    key: str
    value: str | None = None


# FilePart will be used as anonymous knowledge.
class FilePart(BaseModel):
    type: Literal["FilePart"] = Field("FilePart", frozen=True)
    content: str
    # this is not the 'type' field used for polymorphism
    file_type: str = "txt"


class RetrievablePart(BaseModel):
    type: Literal["RetrievablePart"] = Field("RetrievablePart", frozen=True)
    name: str
    tags: list[KnowledgeTag]


# KnowledgePart is a union of FilePart and RetrievablePart
KnowledgePart = Annotated[Union[FilePart, RetrievablePart], Field(discriminator="type")]


class System1Request(BaseModel):
    prompt: str
    model_url: str = Field(..., alias="modelUrl")
    model_family: str = Field(..., alias="modelFamily")
    model_name: str = Field(..., alias="modelName")
    model_key: str = Field(..., alias="modelKey")
    contexts: list[str]
    turns: list[OpenAIMessage]
    collections: Optional[list[KnowledgePart]] = Field(None, alias="collections")
    temperature: float = 0.0
    top_k: int = Field(1, alias="topK")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


@routes.get("/")
async def hello():
    return "Hello, world"


class AgentHome(BaseModel):
    data_path: str = Field(..., description="Base path for data storage")
    org_name: str = Field(..., description="Organization name")

    def __call__(self, agent_name: str) -> str:
        return os.path.join(self.data_path, self.org_name, agent_name)


def get_agent_path(data_path: str, org_name: str, agent_name: str) -> str:
    get_agent_home = AgentHome(data_path=data_path, org_name=org_name)
    return get_agent_home(agent_name)


class InferenceConfig(BaseModel):
    temperature: float = Field(default=0.0, description="temperature")
    topk: int = Field(default=1, description="topk")


@gin.configurable
def get_retriever(agent_path: str, lru_cache: LRU, mode):
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


@routes.get("/index/{org_name}/{agent_name}")
async def check(org_name: str, agent_name: str):
    agent_path = get_agent_path(routes.state.data_path, org_name, agent_name)
    if not os.path.exists(agent_path):
        return JSONResponse(status_code=500, content={"errMsg": "index not found"})
    return JSONResponse(status_code=200, content={})


@routes.post("/index/{org_name}/{agent_name}")
async def build_index_handler(
    org_name: str,
    agent_name: str,
    request: Request,
):
    agent_path = get_agent_path(routes.state.data_path, org_name, agent_name)
    lru_cache = routes.state.retriever_cache

    if os.path.exists(agent_path):
        shutil.rmtree(agent_path)
    os.makedirs(agent_path)

    with tempfile.TemporaryDirectory(prefix="mdrag_") as tmpdirname:
        args = []

        form = await request.form()
        for field_name, field in form.items():
            match field_name:
                case "url":
                    if isinstance(field, str):
                        args.append(field)
                case "tar" | "tar.gz":
                    if isinstance(field, UploadFile):
                        n = tempfile.mkdtemp(dir=tmpdirname)
                        b = io.BytesIO(await field.read())
                        with closing(tarfile.open(fileobj=b)) as t:
                            t.extractall(n)
                        args.append(n)
                case "file":
                    if isinstance(field, UploadFile) and field.filename is not None:
                        _p = os.path.abspath(os.path.join(tmpdirname, field.filename))
                        if not _p.startswith(tmpdirname):
                            return JSONResponse(
                                status_code=500,
                                content={"errMsg": f"file name error {field.filename}"},
                            )

                        os.makedirs(os.path.dirname(_p), exist_ok=True)
                        with tempfile.NamedTemporaryFile(
                            dir=os.path.dirname(_p),
                            suffix=f"_{os.path.basename(_p)}",
                            delete=False,
                        ) as f:
                            f.write(await field.read())
                            args.append(f.name)

        print(args)
        embedding_model = routes.state.embedding_model

        headers = {}
        for k in request.headers.items():
            if len(k) >= 2:
                headers[k[0]] = k[1]
        with open(os.path.join(agent_path, "headers.pickle"), "wb") as f:
            pickle.dump(headers, f)

        build_index(embedding_model, agent_path, *args)
        lru_cache[agent_path] = None

    return JSONResponse(content={})


@routes.post("/v1/tryitnow/")
async def tryitnow(request: Request):
    req = await request.json()

    text = req.get("text", "")
    events = req.get("events", [])
    initial = req.get("initial", "")

    if len(text) == 0:
        return JSONResponse(
            content={"errMsg": "text value is not empty"},
        )

    if not isinstance(initial, bool):
        return JSONResponse(
            content={"errMsg": "initial type is not bool"},
        )

    if not isinstance(events, list):
        return JSONResponse(
            content={"errMsg": "events type is not list"},
        )

    for e in events:
        if not isinstance(e, dict):
            return JSONResponse(
                content={"errMsg": "events[index] type is not dict"},
            )

    return JSONResponse(content={})


@routes.post("/query/{org_name}/{agent_name}")
async def query(org_name: str, agent_name: str, request: Request):
    # create agent home
    agent_home = AgentHome(data_path=routes.state.data_path, org_name=org_name)
    agent_path = agent_home(agent_name)

    backup_prompt = routes.state.prompt

    req = await request.json()
    with open(os.path.join(agent_path, "headers.pickle"), "rb") as f:
        headers: dict = pickle.load(f)

    knowledge_key = req.get("Knowledge-Key") or headers.get("Knowledge-Key".lower())
    knowledge_url = req.get("Knowledge-Url") or headers.get("Knowledge-Url".lower())
    knowledge_model = (
        req.get("Knowledge-Model") or headers.get("Knowledge-Model".lower(), "")
    ).lower()
    knowledge_model_name = req.get("Knowledge-Model-Name") or headers.get(
        "Knowledge-Model-Name".lower()
    )

    if knowledge_model == "":
        knowledge_model = "openai"

    if knowledge_model_name is None:
        logging.info("could not find the model name")
        return JSONResponse(
            content={"errMsg": {"errMsg": "model name found"}},
        )

    knowledge_model_name = f"{knowledge_model}/{knowledge_model_name}"
    logging.info(f"knowledge_model:{knowledge_model}")
    logging.info(f"model_name: {knowledge_model_name}")
    logging.info(f"llm_url: {knowledge_url}")

    if not isinstance(knowledge_url, str):
        return JSONResponse(
            content={"errMsg": {"errMsg": "get Knowledge-Key or Knowledge-Url failed"}},
        )

    generate = Generator(
        agent_home=agent_home,
        app=request.app,
        model_url=knowledge_url,
        model_name=knowledge_model_name,
        model_key=knowledge_key,
    )

    if not os.path.exists(agent_path):
        return JSONResponse(
            content={"errMsg": {"errMsg": "index not found"}},
        )

    # The default does not have tags in the api.
    req["collections"] = [RetrievablePart(name=agent_name, tags=[]).model_dump()]  # type: ignore

    return await generate(req, backup_prompt)


# We assume all the knowledges from the same org is colocated.
@routes.post("/api/v1/{org_name}/generate")
async def generate(org_name: str, request: Request):
    # create agent home
    data_path = routes.state.data_path
    agent_home = AgentHome(data_path=data_path, org_name=org_name)

    req = await request.json()
    logging.info(f"generate request: ${req}")

    model_key = req.get("modelKey")
    model_url = req.get("modelUrl")
    model_family = req.get("modelFamily").lower()
    model_label = req.get("modelName")

    if model_label is None or model_family is None:
        logging.info("could not find the model name")
        return JSONResponse(content={"errMsg": "model name found"})

    model_name = f"{model_family}/{model_label}"
    logging.info(f"model_name: {model_name}")
    logging.info(f"model_url: {model_url}")

    generate = Generator(
        agent_home=agent_home,
        app=request.app,
        model_url=model_url,
        model_name=model_name,
        model_key=model_key,
    )

    return await generate(req, "")


class Generator:
    def __init__(
        self,
        agent_home: AgentHome,
        app: dict,
        model_url: str,
        model_name: str,
        model_key: str | None,
    ):
        self.agent_home = agent_home
        self.template_cache = routes.state.template_cache
        self.lru_cache = routes.state.retriever_cache
        self.model_name = model_name
        self.model_url = model_url
        self.model_key = model_key
        self.adapter = routes.state.adapter

    async def __call__(self, req: dict[str, Any], backup_prompt: str):
        logging.info(f"request: ${req}")
        start_time = time.time()

        turns = req.get("turns", [])
        prompt: str = req.get("prompt", "")

        if len(prompt) == 0:
            prompt = backup_prompt

        if not isinstance(turns, list):
            logging.error("turns is not a list")
            return JSONResponse(content={"errMsg": "turns type is not list"})

        if len(turns) == 0:
            logging.error("empty turns")
            feedback = req.get("feedback", None)
            if feedback:
                return JSONResponse(content={"reply": ""})
            return JSONResponse(content={"errMsg": "turns length cannot be empty"})

        if turns[-1].get("role", "") != "user":
            logging.info("last turn is not from user")
            return JSONResponse(content={"errMsg": "last turn is not from user"})

        user_input = turns[-1].get("content", "")
        req["query"] = user_input

        # We assume the context is what prompt will use,
        if "contexts" not in req:
            collections = req["collections"]

            if isinstance(collections, list) and len(collections) != 0:
                context = []
                for collection_in_json in collections:
                    collection = self.adapter.validate_python(collection_in_json)
                    print(collection)
                    if isinstance(collection, RetrievablePart):
                        print(collection)
                        agent_path = self.agent_home(collection.name)

                        if not os.path.exists(agent_path):
                            return JSONResponse(
                                content={
                                    "errMsg": f"index not found for {collection.name}"
                                }
                            )

                        retriever = get_retriever(agent_path, self.lru_cache)  # type: ignore
                        # What is the result here?
                        context.extend(retriever.retrieve(user_input))
                    elif isinstance(collection, FilePart):
                        context.extend(collection.content)
                    else:
                        return JSONResponse(
                            content={
                                "errMsg": f"do not know how to handle {collection}"
                            }
                        )
                req["context"] = context

        template = get_template(self.template_cache, prompt)

        new_prompt = template.render(**req)
        logging.info("new_prompt")
        logging.info(new_prompt)

        if "temperature" in req:
            temperature = int(req["temperature"])
        else:
            temperature = 0

        llm = get_generator(  # type: ignore
            model=self.model_name,
            api_key=self.model_key,
            model_url=self.model_url,
            temperature=temperature,
        )

        # So that we can use different llm.
        resp = await llm.agenerate(new_prompt, turns)
        logging.info("resp")
        logging.info(resp)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Elapsed time: {elapsed_time}")
        return JSONResponse(content=dataclasses.asdict(resp))


@routes.post("/retrieve/{org_name}/{agent_name}")
async def retrieve(org_name: str, agent_name: str, request: Request):
    agent_path = get_agent_path(routes.state.data_path, org_name, agent_name)
    lru_cache = routes.state.retriever_cache

    req = await request.json()
    user_input = get_user_input(req)

    retriever = get_retriever(agent_path, lru_cache, mode)  # type: ignore

    context = retriever.retrieve(user_input)

    resp = {"reply": context}
    return JSONResponse(content=resp)


class KnowledgeBaseBody(BaseModel):
    query: str
    filter: Optional[List[str]] = None


@routes.post("knowledge_base/{org_name}/{agent_name}")
async def knowledge_base(org_name: str, agent_name: str, item: KnowledgeBaseBody):
    agent_path = get_agent_path(routes.state.data_path, org_name, agent_name)
    lru_cache = routes.state.retriever_cache

    retriever = get_retriever(agent_path, lru_cache)
    context = retriever.retrieve(item.query)

    resp = {"reply": context}
    return JSONResponse(content=resp)


def get_user_input(req: dict[str, Any]):
    turns = req.get("turns", [])
    if len(turns) == 0:
        raise ValueError("the turns that hold user input is not there.")

    if turns[-1].get("role", "") != "user":
        raise ValueError("last turn is not from user")

    return turns[-1].get("content", "")


def get_template(template_cache: LRU, prompt: str) -> Template:
    if len(prompt) == 0:
        raise ValueError("Prompt is missing.")

    if prompt in template_cache:
        template = template_cache[prompt]
    else:
        environment = Environment()
        template = environment.from_string(prompt)
        template_cache[prompt] = template

    return template


def init_app(data_path: str, embedding_model: Any):
    routes.state.data_path = data_path
    routes.state.template_cache = LRU(1024)
    routes.state.retriever_cache = LRU(512)
    routes.state.embedding_model = embedding_model
    routes.state.adapter = TypeAdapter(KnowledgePart)
    routes.state.prompt = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{{context}}"
        "\n---------------------\n"
        "Given this information, please answer the question: {{query}}\n"
    )


if __name__ == "__main__":
    import uvicorn

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

    init_app(p, embedding_model)
    uvicorn.run(routes, host="0.0.0.0", port=8080)
