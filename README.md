## Retrieval augmented generation on markdown

Markdown is a lightweight markup language that allows you to format text in a way that is easy to  read and
write. It's commonly used for creating content that will be displayed on the web, such as in websites,
blogs, forums, and documentation. Markdown is designed to be simple and intuitive, allowing you to use
plain text to add formatting elements like headers, lists, links, images, and more, without the need for
complex HTML or other formatting languages.

Because of the simplicity of its markup language, it is fairly easy to parse the semantic structure
from the markdown file, for example, headers are generally used as title for subsections. These structure
can then be used during both the index and retrieval phase for better performance. MdRag is a simple
retrieval augmented generation system focus on structured semantics document (markup has semantic meaning in
addition to look and feel) such as markdown files.

MdRag also allow you to use different prompt using OpenAI chat like API to build different chat experience,
while bring your private text to LLMs (use llamaindex), simply use the handlebars via pybars3. Note
it is possible to use other LLMs (llama v2 for example) for generation using GenossGPT.


```commandline
pip install -r requirements.txt
// Create index and persist to local disk
python3 rag-index.py output inputs
// With openai as generation, this serves the reg.
python3 reg-serve.py output 
```

