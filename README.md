ETD RAG Pipeline
================

CLI utility for preparing electronic thesis and dissertations for LLM workflow

# Initial Setup

Create the GitHub repository with the GitHub CLI

```
gh repo create kaust-library-systems/etd_rag_pipeline \
--public \
--add-readme \
--description "CLI utility for preparing electronic thesis and dissertations for LLM workflow" \
--license MIT \
--gitignore Python
```

Initialize the UV repository

```
garcm0b@KW20207:~/Work/etd_rag_pipeline$ uv init
Initialized project `etd-rag-pipeline`
garcm0b@KW20207:~/Work/etd_rag_pipeline$
```

> _Note_: I'm not sure if we need to initialize the environment.
Next initialize the virtual environment

```
garcm0b@KW20207:~/Work/etd_rag_pipeline$ uv venv .venv
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
garcm0b@KW20207:~/Work/etd_rag_pipeline$
```

# Setting Up for Usage

# Using the CLI Tool

ChromaDB offers a CLI tool:

```
https://docs.trychroma.com/docs/cli/install
```

# Starting the Server

First start the server with the path to the database as parameter:

```
garcm0b@KW20207:/data/ETD_rag$ chroma run --path ./etd_rag.db/
```

And then connect to the server on another window:

```
garcm0b@KW20207:/data/ETD_rag$ chroma browse "ETD" --local
```

