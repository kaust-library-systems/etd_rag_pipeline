# ETD RAG Pipeline

A semantic search system for the KAUST Electronic Theses and Dissertations (ETD) collection. It allows researchers to query the collection in plain English and retrieve the most relevant theses ranked by meaning, not just keywords.

## Overview

The pipeline ingests Markdown versions of theses into a vector store and exposes a query interface that returns ranked results with author, title, and abstract. The system runs entirely on-premise using [Ollama](https://ollama.com) for embeddings and [ChromaDB](https://www.trychroma.com) as the vector store.

## Current Capabilities

- Ingest Markdown files into a persistent ChromaDB vector store
- Filter out Arabic-language sections during ingestion to reduce noise
- Query the vector store using Maximum Marginal Relevance (MMR) search
- Save query results to CSV with author, title, and abstract fetched live from the KAUST repository API
- Incremental indexing — re-running ingestion only processes new or changed files

## Repository Structure

```
ingest.py       — ingestion pipeline
query.py        — query interface (CLI)
chat_etd.py     — experimental chat interface
utils.py        — shared utilities
```

## Setup

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for environment management
- [Ollama](https://ollama.com) with `granite-embedding:30m` pulled

### Create and activate the virtual environment

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

### Pull the embedding model

```bash
ollama pull granite-embedding:30m
```

## Ingestion

Run the ingestion script to index Markdown files into the vector store:

```bash
python ingest.py
```

### How ingestion works

Each Markdown file is processed in two stages:

1. **Header splitting** — the document is split on Markdown headers (`#`, `##`, `###`) using `MarkdownHeaderTextSplitter`, keeping section titles inside each chunk for better semantic context.
2. **Character splitting** — oversized sections are further split with `RecursiveCharacterTextSplitter` (`chunk_size=1500`, `chunk_overlap=300`).

This structure-aware approach produces semantically coherent chunks (e.g. abstract, introduction, methodology as separate units) rather than arbitrary character-count splits.

### Arabic content filtering

Before indexing, each chunk is scanned for Arabic characters. Chunks where more than 30% of characters are Arabic are dropped. This prevents bilingual abstracts and Arabic-only sections from creating misleading connections between theses. English content from the same document is preserved and indexed normally.

### Embedding model

The system uses `granite-embedding:30m` via Ollama. During development, `granite-embedding:278m` was also tested. The larger model did not produce measurably better results on the benchmark query set and was not adopted.

### Vector store stats (current collection)

- **Documents:** 107 Markdown files
- **Total chunks:** ~16,000
- **Average chunks per document:** ~150

### Paths (currently hard-coded, see roadmap)

| Path | Purpose |
|------|---------|
| `/data/ETD_rag/markdown/` | Input Markdown files |
| `/data/ETD_rag/md_db/etd_rag.db` | ChromaDB vector store |
| `/data/ETD_rag/md_db/record_manager.db` | Incremental index record manager |
| `/data/ETD_rag/metadata.csv` | Thesis metadata (author, title, handle) |

## Querying

```bash
python query.py "water treatment membranes"
python query.py "water treatment membranes" --k 30 --fetch-k 500 --output results.csv
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 50 | Number of final results to return |
| `fetch_k` | 170 | Number of candidate chunks to consider before MMR re-ranking |

### Choosing k and fetch_k

With ~150 chunks per document, a `fetch_k` of 170 only covers 1–2 documents before re-ranking — too narrow for a broad query. Testing showed that `fetch_k=500` improves recall and reduces false positives for broad queries. Recommended defaults are `k=30`, `fetch_k=500`.

### Output

Results are saved to a CSV file with the following fields:

| Field | Description |
|-------|-------------|
| `source` | Path to the Markdown file |
| `author` | Thesis author |
| `title` | Thesis title |
| `abstract` | Full abstract fetched live from the KAUST repository API |

## ChromaDB

The vector store is a persistent ChromaDB instance stored on disk. It can also be browsed via the ChromaDB CLI.

### Browse the vector store locally

Start the ChromaDB server:

```bash
chroma run --path /data/ETD_rag/md_db/etd_rag.db
```

Browse in another terminal:

```bash
chroma browse "ETD" --local
```

### Using Docker

```bash
docker pull chromadb/chroma:1.5.7
docker run -v /data/ETD_rag/etd_rag.db:/data -p 8000:8000 chromadb/chroma:1.5.7
chroma browse "ETD" --host http://localhost:8000
```

## Roadmap

### Near term

- **Configuration file** — replace hard-coded paths with a `ConfigParser` `.ini` file shared between `ingest.py` and `query.py`
- **Streamlit web interface** — single-page application with a search box, example queries, and ranked results displayed in the browser
- **Clickable repository links** — generate a direct URL to each thesis in the KAUST repository from the handle stored in metadata
- **Export formats** — add JSON and HTML export options alongside CSV
- **Query expansion** — use an LLM to rewrite or expand the user's query before searching, available as an opt-in toggle

### Future considerations

- **Chat interface** — conversational search with multi-turn refinement; requires conversation memory management
- **Repository integration** — embed the search directly into the KAUST repository interface; requires infrastructure planning for concurrent users and managed embeddings
- **Multilingual support** — automatic query translation for Arabic-language queries; deferred until Arabic embedding models mature

## License

MIT
