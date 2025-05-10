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
from typing import Optional, Literal, Annotated, Union, List, Dict, Any

import gin
from lru import LRU

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Path, Depends
from fastapi.responses import JSONResponse, Response
from contextlib import closing
from llama_index.core import (
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    set_global_service_context,
)

from jinja2 import Environment
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, BeforeValidator

from rag_index import build_index
from processors.embedding import get_embedding
from processors.llm import get_generator
from processors.retriever import HybridRetriever

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

app = FastAPI()

# --- Configuration and Dependency Management ---

class Settings(BaseModel):
    data_path: str
    template_cache: LRU = Field(default_factory=lambda: LRU(1024))
    retriever_cache: LRU = Field(default_factory=lambda: LRU(512))
    embedding_model: Any
    adapter: TypeAdapter[Any] # Will be initialized at startup
    prompt: str = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query}\n"
    )

    # Pydantic model for settings, with default factories for caches
    class Config:
        arbitrary_types_allowed = True # Allow LRU and TypeAdapter
        extra = "ignore" # Ignore extra fields during validation

settings: Optional[Settings] = None

# Dependency for accessing settings
async def get_settings() -> Settings:
    if settings is None:
        raise RuntimeError("Settings not initialized. Call setup_app_state first.")
    return settings

# --- Pydantic Models ---

class OpenAIMessage(BaseModel):
    role: str
    content: str

class KnowledgeTag(BaseModel):
    key: str
    value: str | None = None

class FilePart(BaseModel):
    type: Literal["FilePart"] = Field("FilePart", frozen=True)
    content: str
    file_type: str = "txt"

class RetrievablePart(BaseModel):
    type: Literal["RetrievablePart"] = Field("RetrievablePart", frozen=True)
    name: str
    tags: list[KnowledgeTag]

KnowledgePart = Annotated[
    Union[FilePart, RetrievablePart],
    Field(discriminator="type")
]

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

class AgentHome(BaseModel):
    data_path: str = Field(..., description="Base path for data storage")
    org_name: str = Field(..., description="Organization name")

    def __call__(self, agent_name: str) -> str:
        return os.path.join(self.data_path, self.org_name, agent_name)

def get_agent_path(org: str, agent_name: str, settings: Settings = Depends(get_settings)) -> str:
    get_agent_home = AgentHome(data_path=settings.data_path, org_name=org)
    return get_agent_home(agent_name)

class InferenceConfig(BaseModel):
    temperature: float = Field(default=0.0, description="temperature")
    topk: int = Field(default=1, description="topk")

@gin.configurable
def get_retriever(agent_path: str, lru_cache: LRU, mode: str = "hybrid"):
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

# --- Endpoints ---

@app.get("/")
async def hello():
    return Response(content="Hello, world")

@app.get("/index/{org}/{agent}")
async def check_index(agent_path: str = Depends(get_agent_path)):
    if not os.path.exists(agent_path):
        raise HTTPException(status_code=500, detail="index not found")
    return JSONResponse(content={})

@app.post("/index/{org}/{agent}")
async def build_index_handler_fastapi(
    request: Request, # Still need Request to get all headers or process multipart
    org: str = Path(...),
    agent: str = Path(...),
    files: List[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    settings: Settings = Depends(get_settings) # Inject settings
):
    agent_path = get_agent_path(org, agent, settings) # Use the dependency for agent_path
    lru_cache = settings.retriever_cache # Access from settings

    if os.path.exists(agent_path):
        shutil.rmtree(agent_path)
    os.makedirs(agent_path)

    args = []
    with tempfile.TemporaryDirectory(prefix="mdrag_") as tmpdirname:
        if url:
            args.append(url)

        # Handle file uploads, including potential tar/tar.gz
        if files:
            for file_upload in files:
                filename = file_upload.filename
                _p = os.path.abspath(os.path.join(tmpdirname, filename))
                if not _p.startswith(tmpdirname):
                    raise HTTPException(status_code=400, detail=f"File name error: {filename}")

                os.makedirs(os.path.dirname(_p), exist_ok=True)
                file_content = await file_upload.read()

                if filename.endswith((".tar", ".tar.gz")):
                    try:
                        with closing(tarfile.open(fileobj=io.BytesIO(file_content))) as t:
                            t.extractall(tmpdirname) # Extract to tmpdirname, args will point to that dir
                        args.append(tmpdirname) # Add the directory containing extracted files
                    except tarfile.ReadError as e:
                        raise HTTPException(status_code=400, detail=f"Error extracting tar file {filename}: {e}")
                else:
                    with open(_p, "wb") as f:
                        f.write(file_content)
                    args.append(_p) # Add individual file path
                await file_upload.close()

        logging.info(f"Processed args for build_index: {args}")
        embedding_model = settings.embedding_model # Access from settings

        headers = dict(request.headers) # Get all headers
        with open(os.path.join(agent_path, "headers.pickle"), "wb") as f:
            pickle.dump(headers, f)

        build_index(embedding_model, agent_path, *args)
        if agent_path in lru_cache:
             lru_cache[agent_path] = None # Invalidate cache

    return JSONResponse(content={})

class TryItNowRequest(BaseModel):
    text: str
    events: List[Dict[str, Any]] = []
    initial: bool = False

@app.post("/v1/tryitnow/")
async def tryitnow(payload: TryItNowRequest):
    if not payload.text: # Simpler check for empty string
        raise HTTPException(status_code=400, detail="Text value cannot be empty")
    # Pydantic models handle type validation automatically
    return JSONResponse(content={})

# Helper function for common header/payload parsing
def get_knowledge_params(req_dict: Dict[str, Any], headers: Dict[str, str]):
    knowledge_key = req_dict.get("Knowledge-Key") or headers.get("knowledge-key")
    knowledge_url = req_dict.get("Knowledge-Url") or headers.get("knowledge-url")
    knowledge_model = (req_dict.get("Knowledge-Model") or headers.get("knowledge-model", "")).lower()
    knowledge_model_name = req_dict.get("Knowledge-Model-Name") or headers.get("knowledge-model-name")

    if not knowledge_model:
        knowledge_model = "openai"

    if not knowledge_model_name:
        logging.info("Could not find the model name")
        raise HTTPException(status_code=400, detail="Model name not found")

    resolved_knowledge_model_name = f"{knowledge_model}/{knowledge_model_name}"
    logging.info(f"Knowledge_model:{knowledge_model}")
    logging.info(f"Model_name: {resolved_knowledge_model_name}")
    logging.info(f"LLM_url: {knowledge_url}")

    return knowledge_key, knowledge_url, resolved_knowledge_model_name


class QueryRequest(BaseModel):
    Knowledge_Key: Optional[str] = Field(None, alias="Knowledge-Key")
    Knowledge_Url: Optional[str] = Field(None, alias="Knowledge-Url")
    Knowledge_Model: Optional[str] = Field(None, alias="Knowledge-Model")
    Knowledge_Model_Name: Optional[str] = Field(None, alias="Knowledge-Model-Name")
    turns: List[OpenAIMessage]
    prompt: Optional[str] = None
    feedback: Optional[Any] = None
    collections: Optional[List[Dict[str, Any]]] = None # Raw dicts, validated by adapter
    temperature: Optional[float] = None

@app.post("/query/{org}/{agent}")
async def query(
    request: Request, # Keep for headers
    payload: QueryRequest,
    org: str = Path(...),
    agent: str = Path(...),
    settings: Settings = Depends(get_settings)
):
    agent_path = get_agent_path(org, agent, settings) # Use dependency

    backup_prompt = settings.prompt # Access from settings
    req_dict = payload.model_dump(by_alias=True)

    if not os.path.exists(agent_path):
        raise HTTPException(status_code=404, detail="Index not found")

    try:
        with open(os.path.join(agent_path, "headers.pickle"), "rb") as f:
            stored_headers: dict = pickle.load(f)
    except FileNotFoundError:
        stored_headers = {}

    # Prioritize request payload, then stored headers, then current request headers
    current_request_headers = dict(request.headers)
    all_headers = {**stored_headers, **current_request_headers} # Merge in order of preference

    knowledge_key, knowledge_url, resolved_knowledge_model_name = get_knowledge_params(req_dict, all_headers)

    generator_instance = Generator(
        agent_home=AgentHome(data_path=settings.data_path, org_name=org),
        model_url=knowledge_url,
        model_name=resolved_knowledge_model_name,
        model_key=knowledge_key,
        settings=settings # Pass settings to Generator
    )

    if not req_dict.get("collections"):
        req_dict["collections"] = [RetrievablePart(name=agent, tags=[]).model_dump(by_alias=True)] # Use agent as collection name

    return await generator_instance(req_dict, backup_prompt)

class GenerateApiRequest(BaseModel):
    modelKey: str
    modelUrl: str
    modelFamily: str
    modelName: str
    turns: List[OpenAIMessage]
    prompt: Optional[str] = None
    feedback: Optional[Any] = None
    collections: Optional[List[Dict[str, Any]]] = None
    contexts: Optional[List[str]] = None
    temperature: Optional[float] = None

@app.post("/api/v1/{org}/generate")
async def generate_api(
    request: Request, # Keep for headers
    payload: GenerateApiRequest,
    org: str = Path(...),
    settings: Settings = Depends(get_settings)
):
    req_dict = payload.model_dump(by_alias=True)
    logging.info(f"Generate request: {req_dict}")

    model_key = req_dict.get("modelKey") or request.headers.get("model-key")
    model_url = req_dict.get("modelUrl") or request.headers.get("model-url")
    model_family_req = (req_dict.get("modelFamily") or request.headers.get("model-family", "")).lower()
    model_label = req_dict.get("modelName") or request.headers.get("model-name")

    if not model_label or not model_family_req:
        logging.info("Could not find the model name or family")
        raise HTTPException(status_code=400, detail="Model name or family not found")

    model_name = f"{model_family_req}/{model_label}"
    logging.info(f"Model_name: {model_name}")
    logging.info(f"Model_url: {model_url}")

    generator_instance = Generator(
        agent_home=AgentHome(data_path=settings.data_path, org_name=org),
        model_url=model_url,
        model_name=model_name,
        model_key=model_key,
        settings=settings # Pass settings to Generator
    )

    return await generator_instance(req_dict, None)

class Generator:
    def __init__(self, agent_home: AgentHome, model_url: str, model_name: str, model_key: str, settings: Settings):
        self.agent_home = agent_home
        self.template_cache = settings.template_cache
        self.lru_cache = settings.retriever_cache
        self.model_name = model_name
        self.model_url = model_url
        self.model_key = model_key
        self.adapter = settings.adapter

    async def __call__(self,  req: dict[str, Any], backup_prompt: Optional[str] = None):
        logging.info(f"Generator request: {req}")
        start_time = time.time()

        turns = req.get("turns", [])
        prompt_from_req = req.get("prompt", "")

        final_prompt_str = prompt_from_req
        if not final_prompt_str and backup_prompt:
            final_prompt_str = backup_prompt

        if not isinstance(turns, list) or not turns:
            logging.error("Turns list is empty or invalid.")
            raise HTTPException(status_code=400, detail="Turns list cannot be empty or invalid.")

        if turns[-1].get("role", "") != "user":
            logging.info("Last turn is not from user")
            raise HTTPException(status_code=400, detail="Last turn is not from user")

        user_input = turns[-1].get("content", "")
        req["query"] = user_input

        if not req.get("contexts"):
            collections = req.get("collections")
            if isinstance(collections, list) and collections:
                context_nodes = []
                for collection_in_json in collections:
                    if not isinstance(collection_in_json, dict):
                        raise HTTPException(status_code=400, detail=f"Invalid collection item: {collection_in_json}")
                    try:
                        collection = self.adapter.validate_python(collection_in_json)
                    except ValidationError as e: # More specific exception
                        logging.error(f"Pydantic validation error for collection: {e}")
                        raise HTTPException(status_code=400, detail=f"Invalid collection structure: {collection_in_json}. Error: {e}")

                    logging.info(f"Processing collection: {collection}")
                    if isinstance(collection, RetrievablePart):
                        logging.info(f"RetrievablePart: {collection}")
                        agent_path = self.agent_home(collection.name)

                        if not os.path.exists(agent_path):
                            raise HTTPException(status_code=404, detail=f"Index not found for {collection.name}")

                        retriever_instance = get_retriever(agent_path, self.lru_cache)  # type: ignore
                        if retriever_instance is None:
                            raise HTTPException(status_code=500, detail=f"Could not initialize retriever for {collection.name}")
                        retrieved_items = retriever_instance.retrieve(user_input)
                        for item in retrieved_items:
                             if hasattr(item, 'node') and hasattr(item.node, 'text'):
                                 context_nodes.append(item.node.text)
                             elif hasattr(item, 'text'):
                                 context_nodes.append(item.text)
                             else:
                                 logging.warning(f"Retrieved item does not have expected text attribute: {item}")
                    elif isinstance(collection, FilePart):
                        context_nodes.append(collection.content)
                    else:
                        raise HTTPException(status_code=400, detail=f"Do not know how to handle {collection}")
                req["context"] = "\n---\n".join(context_nodes)
            else:
                req["context"] = ""
        else:
            if isinstance(req["contexts"], list):
                req["context"] = "\n---\n".join(req["contexts"])
            else:
                req["context"] = str(req["contexts"])

        if not final_prompt_str:
            logging.error("Prompt string is empty before rendering.")
            raise HTTPException(status_code=500, detail="Prompt configuration error: prompt string is empty.")

        template = get_template(self.template_cache, final_prompt_str)

        template_data = {
            "context": req.get("context", ""),
            "query": req.get("query", ""),
            **req
        }
        new_prompt_rendered = template.render(**template_data)
        logging.info("Rendered new_prompt:")
        logging.info(new_prompt_rendered)

        temperature_val = req.get("temperature")
        if temperature_val is not None:
            try:
                temperature_val = float(temperature_val)
            except (ValueError, TypeError):
                logging.warning(f"Invalid temperature value: {temperature_val}, defaulting to 0.0")
                temperature_val = 0.0
        else:
            temperature_val = 0.0

        llm = get_generator(  # type: ignore
            model=self.model_name,
            api_key=self.model_key,
            model_url=self.model_url,
            temperature=temperature_val
        )

        resp = await llm.agenerate(new_prompt_rendered, turns)
        logging.info(f"LLM resp: {resp}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Elapsed time: {elapsed_time}")
        return JSONResponse(content=dataclasses.asdict(resp))

class RetrieveRequest(BaseModel):
    turns: List[OpenAIMessage]
    mode: Optional[str] = "hybrid"

@app.post("/retrieve/{org}/{agent}")
async def retrieve(
    payload: RetrieveRequest,
    org: str = Path(...),
    agent: str = Path(...),
    settings: Settings = Depends(get_settings)
):
    agent_path = get_agent_path(org, agent, settings) # Use dependency
    lru_cache = settings.retriever_cache # Access from settings

    req_dict = payload.model_dump()

    try:
        user_input = get_user_input(req_dict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    mode_to_use = payload.mode if payload.mode else "hybrid"

    retriever_instance = get_retriever(agent_path, lru_cache, mode_to_use)  # type: ignore
    if retriever_instance is None:
        raise HTTPException(status_code=500, detail=f"Could not initialize retriever for agent {agent} with mode {mode_to_use}")

    context_nodes = retriever_instance.retrieve(user_input)
    extracted_texts = []
    for item in context_nodes:
        if hasattr(item, 'node') and hasattr(item.node, 'text'):
            extracted_texts.append(item.node.text)
        elif hasattr(item, 'text'):
            extracted_texts.append(item.text)
        else:
            logging.warning(f"Retrieved item in /retrieve endpoint does not have expected text attribute: {item}")

    resp_content = {"reply": extracted_texts}
    return JSONResponse(content=resp_content)

def get_user_input(req: dict[str, Any]):
    turns = req.get("turns", [])
    if not turns or not isinstance(turns, list):
        raise ValueError("The turns that hold user input is not there or invalid.")
    if turns[-1].get("role", "") != "user":
        raise ValueError("Last turn is not from user")
    content = turns[-1].get("content", "")
    if not isinstance(content, str):
        raise ValueError("User input content must be a string.")
    return content

def get_template(template_cache: LRU, prompt: str) -> Any:
    if not prompt: # Simpler check for empty string
        raise ValueError("Prompt is missing.")
    if prompt in template_cache:
        template = template_cache[prompt]
    else:
        environment = Environment()
        try:
            template = environment.from_string(prompt)
            template_cache[prompt] = template
        except Exception as e:
            logging.error(f"Error compiling Jinja2 template: {e}\nTemplate content: {prompt}")
            raise ValueError(f"Invalid template syntax: {e}")
    return template

def setup_app_state(data_path_val: str, embedding_model_val: Any):
    global settings
    settings = Settings(
        data_path=data_path_val,
        embedding_model=embedding_model_val,
        adapter=TypeAdapter(KnowledgePart) # Initialize adapter here
    )

if __name__ == "__main__":
    import uvicorn

    gin.parse_config_file("serve.gin")
    embedding = get_embedding() # type: ignore

    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <data_path_directory>")
        sys.exit(1)

    p_datapath = sys.argv[1]
    if not os.path.isdir(p_datapath):
        print(f"Error: Data path '{p_datapath}' is not a directory or does not exist.")
        sys.exit(1)

    setup_app_state(p_datapath, embedding)

    service_context = ServiceContext.from_defaults(
        llm=None,
        llm_predictor=None,
        embed_model=embedding,
    )
    set_global_service_context(service_context)

    uvicorn.run(app, host="0.0.0.0", port=8000)
