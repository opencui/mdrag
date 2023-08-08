#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import logging
from langchain.embeddings import HuggingFaceEmbeddings

from llama_index import ServiceContext
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import LangchainEmbedding
from llama_index import set_global_service_context

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# init download hugging fact model
langchain_embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_kwargs={'device': 'cpu'},
)
embed_model = LangchainEmbedding(langchain_embedding)
service_context = ServiceContext.from_defaults(embed_model=embed_model)
set_global_service_context(service_context)

# python rag-index doc_path index_path
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)

    p1, p2 = sys.argv[1], sys.argv[2]

    if os.path.isfile(p1):
        documents = SimpleDirectoryReader(input_files=[p1]).load_data()
    elif os.path.isdir(p1):
        documents = SimpleDirectoryReader(input_dir=p1).load_data()
    else:
        sys.exit(1)

    try:
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=p2)
    except Exception as e:
        print(str(e))
        shutil.rmtree(p2, ignore_errors=True)
