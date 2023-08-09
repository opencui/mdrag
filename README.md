## Retrieval augmented generation as System 1

To build cost effective conversational experience, it is useful to take advantage of every things you had, and retrieval augmented generation is a great way to field informational query using the existing material you build along the way. Aiming for being System 1, this project assumes that the dependibility is created on top of the LLMs based RAGs, so that RAGs itself can stay as simply as possible. In particular, we focus on being easy to deploy, make indexing phase extensible, and providing an universal OpenAI base chat interface regardless which LLMs you use for generation (thanks to GenossGPT).

System 1 is assumed to be language dependent, and it requires access to two different models: one for creating embedding for qeury and nodes, another for generating response based on prompt and context from the retrieved context. Generally we want a bigger model for generation.

Embedding: For English, we use the following sentence-transformer package, and in particular we use the multi-qa-mpnet-base-dot-v1 as default.


### Install

- pip install -r requirements.txt

### Test Command

- OPENAI_API_KEY="" pytest test.py

### Run Command

- OPENAI_API_KEY="" python main.py
