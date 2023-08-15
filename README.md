## Retrieval augmented generation as System 1

RAG system 1 allow you can different prompt with OpenAI chat API to build different chat experience,
while bring your private text to LLMs (use llamaindex), simply use the handlebars via pybars3. Note
it is possible to use other LLMs (llama v2 for example) using GenossGPT.

To quality of the RAG system depends on both retrieval and generation. The retrieval quality are decided 
by how documents are parsed, and how nodes are combined before they are sent into LLMs, which will be the
focus of this project.



### Install

- pip install -r requirements.txt

### Test Command

- OPENAI_API_KEY="" pytest test.py

### Run Command

- OPENAI_API_KEY="" python main.py
