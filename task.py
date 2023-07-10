import functools
import io
import os
import uuid
import shutil
import asyncio
import zipfile
import tarfile
import tempfile

from os import path
from functools import lru_cache
from aiohttp import ClientSession
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from llama_index.response.schema import RESPONSE_TYPE
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage

lock = Lock()
executor = ThreadPoolExecutor(max_workers=20)
tmp_path_prefix = "upload-path-"
WORKSPACE = "./workspace/"
if not path.isdir(WORKSPACE):
    os.mkdir(WORKSPACE)


def text_func(s) -> str:
    tmp_path = tempfile.mkdtemp(prefix=tmp_path_prefix)
    with open(f'{tmp_path}/{uuid.uuid4().hex}', 'w') as f:
        f.write(s)
    return tmp_path


def zip_func(io_buf) -> str:
    tmp_path = tempfile.mkdtemp(prefix=tmp_path_prefix)
    with zipfile.ZipFile(io_buf, "r") as z:
        z.extractall(tmp_path)
    return tmp_path


def tar_func(io_buf) -> str:
    tmp_path = tempfile.mkdtemp(prefix=tmp_path_prefix)
    with tarfile.open(fileobj=io_buf) as tar:
        tar.extractall(tmp_path)
    return tmp_path


async def url_text_func(url) -> str:
    async with ClientSession(trust_env=True) as client:
        async with client.get(url) as resp:
            body = await resp.text()
            return text_func(body)


async def url_tar_func(url) -> str:
    async with ClientSession(trust_env=True) as client:
        async with client.get(url) as resp:
            body = await resp.read()
            return tar_func(io.BytesIO(body))


async def url_zip_func(url) -> str:
    async with ClientSession(trust_env=True) as client:
        async with client.get(url) as resp:
            body = await resp.read()
            return zip_func(io.BytesIO(body))


task_init_map_func = {
    "tar": tar_func,
    "zip": zip_func,
    "text": text_func,
    "url+tar": url_tar_func,
    "url+zip": url_zip_func,
    "url+text": url_text_func,
}


def get_index_path(uid):
    return path.join(WORKSPACE, uid)


@lru_cache(maxsize=10240)
def task_query(uid, input) -> RESPONSE_TYPE:
    index_path = get_index_path(uid)

    if not path.isdir(index_path):
        raise Exception("task query {uid} index path not found")

    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(use_async=True)
    resp = query_engine.query(input)
    return resp


def run_on_executor(fn, *args, **kwargs):
    return asyncio.get_event_loop().run_in_executor(
        executor, functools.partial(fn, *args, **kwargs))


def _task_build_index(docs_path, index_path):
    documents = SimpleDirectoryReader(docs_path, recursive=True).load_data()
    index = VectorStoreIndex.from_documents(documents)
    try:
        with lock:
            if not path.isdir(index_path):
                index.storage_context.persist(persist_dir=index_path)
    except:
        if path.isdir(index_path):
            shutil.rmtree(index_path, ignore_errors=True)

    if path.isdir(docs_path):
        shutil.rmtree(docs_path, ignore_errors=True)


async def task_build_index(fileType, uid, fileData):
    index_path = get_index_path(uid)

    if not path.isdir(index_path):
        init_task = task_init_map_func.get(fileType)
        if init_task == None:
            raise Exception(f'init task {fileType} not found')

        try:
            if 'url+' not in fileType:
                docs_path = init_task(fileData)
            else:
                docs_path = await init_task(fileData)
        except:
            raise Exception(f'init task run failed')

        await run_on_executor(_task_build_index, docs_path, index_path)
