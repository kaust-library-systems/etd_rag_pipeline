ETD RAG Pipeline
================

CLI utility for preparing electronic thesis and dissertations for LLM workflow

# Set up

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

Next initialize the virtual environment

```
garcm0b@KW20207:~/Work/etd_rag_pipeline$ uv venv .venv
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
garcm0b@KW20207:~/Work/etd_rag_pipeline$
```
