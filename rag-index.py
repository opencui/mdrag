#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import logging

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import ServiceContext
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import set_global_service_context
from processors.markdown import MarkdownReader
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    EntityExtractor,
    MetadataFeatureExtractor,
)
from pathlib import Path

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# init download hugging fact model
service_context = ServiceContext.from_defaults(
    llm_predictor=None,
    llm=None,
    embed_model=HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_kwargs={'device': 'cpu'}))

set_global_service_context(service_context)


# python rag-index doc_path index_path
if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)

    # We assume that there output directory is the first argument, and the rest is input directory
    p2 = sys.argv[1]

    documents = []
    for p1 in sys.argv[2:]:
        if os.path.isfile(p1) and p1.endswith(".md"):
            documents.extend(MarkdownReader().load_data(Path(p1)))
        elif os.path.isdir(p1):
            documents.extend(
                SimpleDirectoryReader(
                    input_dir=p1,
                    exclude=["*.rst", "*.ipynb", "*.py", "*.bat", "*.txt", "*.png", "*.jpg", "*.jpeg", "*.csv", "*.html",
                             "*.js", "*.css", "*.pdf", "*.json"],
                    file_extractor={".md": MarkdownReader()},
                    recursive=True,
                ).load_data())

    # exclude these things from considerations.
    for doc in documents:
        doc.excluded_llm_metadata_keys = ["file_name", "content_type"]
        doc.excluded_embed_metadata_keys = ["file_name", "content_type"]

    try:
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=p2)
    except Exception as e:
        print(str(e))
        shutil.rmtree(p2, ignore_errors=True)
