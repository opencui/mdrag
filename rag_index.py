#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import os.path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any
from pathlib import Path
from urllib.parse import urlparse

import gin
import requests
from llama_index.core import (
    ServiceContext,
    SimpleDirectoryReader,
    SimpleKeywordTableIndex,
    StorageContext,
    VectorStoreIndex,
    set_global_service_context,
)

from processors.embedding import get_embedding
from processors.markdown import MarkdownReader


sdr_exclude = [
    "*.rst",
    "*.ipynb",
    "*.py",
    "*.bat",
    "*.txt",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.csv",
    "*.html",
    "*.js",
    "*.css",
    "*.pdf",
    "*.json",
]

re_github = r"https://(?P<token>.*?)github\.com/(?P<org>[^/]+)/(?P<repo>[^/\s]+)/?(?P<type>(tree|blob)/?(?P<version>[^/\s]+)/?(?P<path>.+)?)?"


def file_reader(file_path: str):
    return MarkdownReader().load_data(Path(file_path))


def dir_reader(dir_path: str):
    return SimpleDirectoryReader(
        input_dir=dir_path,
        exclude=sdr_exclude,
        file_extractor={".md": MarkdownReader()},
        recursive=True,
    ).load_data()


def url_reader(url: str):
    logging.info(f"{url} start")
    resp = requests.get(url, timeout=300)
    if "text/" in resp.headers.get("content-type", ""):
        f = tempfile.NamedTemporaryFile(suffix=".md", delete=False)
        f.write(resp.content)
        f.close()
        docs = file_reader(f.name)
        shutil.rmtree(f.name, ignore_errors=True)
        return docs
    return []


def github_reader(urlParse: re.Match):
    urlReGroups = urlParse.groups()
    token = urlReGroups[0]
    org = urlReGroups[1]
    repo = urlReGroups[2]
    version = urlReGroups[4]  # None|tree|blob
    branch = urlReGroups[5]  # tag_name|branch_name|commit_id
    # version == tree, path is dir; version == blob, path is file
    sub_path = "" if urlReGroups[6] is None else urlReGroups[6]

    if version == "blob":
        url = (
            f"https://{token}raw.githubusercontent.com/{org}/{repo}/{branch}/{sub_path}"
        )
        return url_reader(url)

    if version not in [None, "tree"]:
        return []

    url = f"https://{token}github.com/{org}/{repo}"

    if branch:
        args = ["git", "clone", "--depth", "1", "--branch", branch, url, "."]
    else:
        args = ["git", "clone", "--depth", "1", url, "."]

    del_not_md = """find . -type f ! -name "*.md" | xargs rm -rf"""
    logging.info(f"{args} start")
    with tempfile.TemporaryDirectory() as tmpdirname:
        subprocess.run(args, check=True, timeout=300, cwd=tmpdirname)
        subprocess.run(del_not_md, shell=True, timeout=300, cwd=tmpdirname)
        logging.info(f"{args} ended")
        docs = dir_reader(os.path.join(tmpdirname, sub_path))
        return docs


map_func = {
    "dir": dir_reader,
    "file": file_reader,
    "github": github_reader,
    "url": url_reader,
}


def build_index(embed_model: Any, output: str, *args: str):
    gin.parse_config_file("index.gin")

    service_context = ServiceContext.from_defaults(
        llm=None, llm_predictor=None, embed_model=embed_model
    )

    storage_context = StorageContext.from_defaults()
    set_global_service_context(service_context)

    documents = []
    for file_path in args:
        if os.path.isfile(file_path) and file_path.endswith(".md"):
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

    for doc in documents:
        doc.excluded_llm_metadata_keys = ["file_name", "content_type"]
        doc.excluded_embed_metadata_keys = ["file_name", "content_type"]

    try:
        embedding_index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        keyword_index = SimpleKeywordTableIndex(
            documents, storage_context=storage_context
        )

        embedding_index.set_index_id("embedding")
        embedding_index.storage_context.persist(persist_dir=output)
        keyword_index.set_index_id("keyword")
        keyword_index.storage_context.persist(persist_dir=output)
    except Exception as e:
        print(str(e))
        shutil.rmtree(output, ignore_errors=True)


# python rag-index index_persist_path collection_path...
# collection_path
#     data/
#     /data
#     https://abc.com/xyz.md
#     https://<token>@github.com/<org>/<repo>
#     https://<token>@github.com/<org>/<repo>/tree/<tag_name|branch_name>/<sub_dir>
#     https://<token>@github.com/<org>/<repo>/blob/<tag_name|branch_name|commit_id>/<sub_dir>/<file_name>.md
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    gin.parse_config_file("index.gin")

    if len(sys.argv) < 3:
        sys.exit(0)

    model = get_embedding()  # type: ignore
    build_index(model, sys.argv[1], *sys.argv[2:])
