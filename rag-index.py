#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import gin
import shutil
import logging
import tempfile
import requests
import subprocess

from pathlib import Path
from urllib.parse import urlparse

from llama_index import ServiceContext, StorageContext
from llama_index import VectorStoreIndex, SimpleDirectoryReader, SimpleKeywordTableIndex
from llama_index import set_global_service_context
from processors.markdown import MarkdownReader
from processors.embedding import get_embedding

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

sdr_exclude = [
    "*.rst", "*.ipynb", "*.py", "*.bat", "*.txt", "*.png", "*.jpg", "*.jpeg",
    "*.csv", "*.html", "*.js", "*.css", "*.pdf", "*.json"
]

re_github = r"https://(?P<token>[^@]*?)@?github\.com/(?P<org>.*?)/(?P<repo>[^/@]+)@?(?P<branch>\S+)?/?"


def file_reader(file_path: str):
    return MarkdownReader().load_data(Path(file_path))


def dir_reader(dir_path: str):
    return SimpleDirectoryReader(
        input_dir=dir_path,
        exclude=sdr_exclude,
        file_extractor={
            ".md": MarkdownReader()
        },
        recursive=True,
    ).load_data()


def url_reader(url: str):
    resp = requests.get(url, timeout=300)
    if "text/" in resp.headers.get('content-type', ""):
        f = tempfile.NamedTemporaryFile(suffix=".md", delete=False)
        f.write(resp.content)
        f.close()
        docs = file_reader(f.name)
        shutil.rmtree(f.name, ignore_errors=True)
        return docs
    return []


def github_reader(urlParse: re.Match):
    url = urlParse.string
    branch = urlParse.groups()[3]
    if branch:
        url = url.replace(f'@{branch}', "")
        args = ["git", "clone", "--depth", "1", "--branch", branch, url, "."]
    else:
        args = ["git", "clone", "--depth", "1", url, "."]

    del_not_md = '''find . -type f ! -name "*.md" | xargs rm -rf'''
    logging.info(f"{args} start")
    with tempfile.TemporaryDirectory() as tmpdirname:
        subprocess.run(args, check=True, timeout=300, cwd=tmpdirname)
        subprocess.run(del_not_md, shell=True, timeout=300, cwd=tmpdirname)
        print(tmpdirname)
        logging.info(f"{args} ended")
        docs = dir_reader(tmpdirname)
        return docs


map_func = {
    "dir": dir_reader,
    "file": file_reader,
    "github": github_reader,
    "url": url_reader,
}

# python rag-index index_persist_path collection_path...
# collection_path
#     data/
#     /data
#     https://abc.com/xyz.md
#     https://github.com/abc/xyz
#     https://github.com/abc/xyz@branch_name
#     https://github.com/abc/xyz@tag_name
#     https://token@github.com/abc/xyz
if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(0)

    # We assume that there output directory is the first argument, and the rest is input directory
    output = sys.argv[1]
    gin.parse_config_file('index.gin')

    # init download hugging fact model
    service_context = ServiceContext.from_defaults(
        llm=None,
        llm_predictor=None,
        embed_model=get_embedding(),
    )

    storage_context = StorageContext.from_defaults()

    set_global_service_context(service_context)

    documents = []
    for file_path in sys.argv[2:]:
        if os.path.isfile(file_path) and file_path.endswith(".md"):
            print(map_func["file"])
            documents.extend(map_func["file"](file_path))
        elif os.path.isdir(file_path):
            documents.extend(map_func["dir"](file_path))
        else:
            match_github = re.search(re_github, file_path)
            if match_github:
                documents.extend(map_func["github"](match_github))
                continue

            match_url = urlparse(file_path)
            if match_url.scheme and match_url.netloc:
                documents.extend(map_func["url"](file_path))
                continue

    # exclude these things from considerations.
    for doc in documents:
        doc.excluded_llm_metadata_keys = ["file_name", "content_type"]
        doc.excluded_embed_metadata_keys = ["file_name", "content_type"]

    try:
        embedding_index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context)
        keyword_index = SimpleKeywordTableIndex(
            documents, storage_context=storage_context)

        embedding_index.set_index_id("embedding")
        embedding_index.storage_context.persist(persist_dir=output)
        keyword_index.set_index_id("keyword")
        keyword_index.storage_context.persist(persist_dir=output)
    except Exception as e:
        print(str(e))
        shutil.rmtree(output, ignore_errors=True)
