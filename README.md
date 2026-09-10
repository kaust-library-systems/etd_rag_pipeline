# ETD RAG Pipeline

A semantic search system for the KAUST Electronic Theses and Dissertations (ETD) collection. It allows researchers to query the collection in plain English and retrieve the most relevant theses ranked by meaning, not just keywords.

## Overview

The pipeline queries a ChromaDB vector store of thesis chunks (the `etd_pillars`
collection) and returns ranked results with author, title, matching section, and
an abstract fetched live from the KAUST repository API. The system runs entirely
on-premise using [Ollama](https://ollama.com) for embeddings and
[ChromaDB](https://www.trychroma.com) (client-server) as the vector store.

The `etd_pillars` collection is built and maintained by a separate ingestion
project. Every chunk carries full bibliographic metadata (`title`, `author`,
`advisor`, `year`, `handle`, `type`, `department`, `discipline`, `degree_name`,
`section`), so the query side needs no external metadata file.

## Current Capabilities

- Query the vector store using Maximum Marginal Relevance (MMR) search
- Deduplicate chunk hits down to distinct documents, keeping the best-ranked hit
- Save query results to CSV, or browse them in a Streamlit web app
- Abstracts fetched live from the KAUST repository API by handle
- `ingest.py` retained for the legacy Markdown ingestion path (incremental,
  Arabic-section filtering)

## Repository Structure

```
app.py          — Streamlit web interface
query.py        — query interface (CLI) and the shared `search()` function
chat_etd.py     — experimental chat interface
ingest.py       — legacy Markdown ingestion pipeline
utils.py        — shared utilities (used by ingest.py)
config.ini      — paths, ChromaDB host/port/collection, embedding model, k/fetch_k
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

## Web app

```bash
streamlit run app.py
```

A search box with example queries; results show title (linked to the KAUST
repository), author, type, year, department, matching section, and abstract, with
a CSV export button.

## Ingestion (legacy Markdown path)

> The current `etd_pillars` collection is produced by a separate project. The
> script below documents the original in-repo Markdown pipeline and is kept for
> reference.

Run the ingestion script to index Markdown files into the vector store:

```bash
python ingest.py
```

### How ingestion works

Each Markdown file is processed in two stages:

1. **Header splitting** — the document is split on Markdown headers (`#`, `##`, `###`) using `MarkdownHeaderTextSplitter`, keeping section titles inside each chunk for better semantic context.
2. **Character splitting** — oversized sections are further split with `RecursiveCharacterTextSplitter` (`chunk_size=800`, `chunk_overlap=150`).
3. **Section filtering** — non-content sections (references, acknowledgements, table of contents, copyright) are dropped.

This structure-aware approach produces semantically coherent chunks (e.g. abstract, introduction, methodology as separate units) rather than arbitrary character-count splits.

### Arabic content filtering

Before indexing, each chunk is scanned for Arabic characters. Chunks where more than 30% of characters are Arabic are dropped. This prevents bilingual abstracts and Arabic-only sections from creating misleading connections between theses. English content from the same document is preserved and indexed normally.

### Embedding model

The system uses `granite-embedding:30m` via Ollama. During development, `granite-embedding:278m` was also tested. The larger model did not produce measurably better results on the benchmark query set and was not adopted.

### Vector store stats (`etd_pillars` collection)

- **Total chunks:** ~1.24 million
- **Documents:** order of 5,000 theses/dissertations
- **Average chunks per document:** ~220
- **Embedding dimension:** 384 (`granite-embedding:30m`), l2 distance

## Configuration

All runtime settings live in `config.ini`:

```ini
[chromadb]
host = localhost
port = 8000
collection = etd_pillars

[embeddings]
model = granite-embedding:30m

[query]
k = 30
fetch_k = 500
```

## Querying

```bash
python query.py "water treatment membranes"
```

Results are written to `<output_dir>/results.csv`. `k` and `fetch_k` come from
`config.ini`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 30 | Number of final chunk hits to return (deduplicated to distinct documents) |
| `fetch_k` | 500 | Number of candidate chunks to consider before MMR re-ranking |

### Choosing k and fetch_k

With ~220 chunks per document, `fetch_k` needs to be well above a single
document's chunk count for a broad query to reach several distinct theses.
`k=30`, `fetch_k=500` is the current default; revisit against a benchmark query
set as the collection grows.

### Output

CSV fields: `handle`, `url`, `title`, `author`, `type`, `year`, `advisor`,
`department`, `section` (matching chunk heading), `abstract` (fetched live from
the KAUST repository API).

## ChromaDB

The vector store runs as a ChromaDB server. `query.py` and `app.py` connect to it
over HTTP using the host/port in `config.ini`.

### Browse the collection

```bash
chroma browse "etd_pillars" --host http://localhost:8000
```

### Run the server with Docker Compose

```bash
docker compose up -d chromadb
```

This serves the `etd_pillars` database (see `docker-compose.yml` for the mounted
path).

## Roadmap

### Near term

- **Export formats** — add JSON and HTML export options alongside CSV
- **Query expansion** — use an LLM to rewrite or expand the user's query before searching, available as an opt-in toggle
- **Parameter tuning** — re-evaluate `k` / `fetch_k` against a benchmark query set at the full collection scale

### Future considerations

- **Chat interface** — conversational search with multi-turn refinement; requires conversation memory management
- **Repository integration** — embed the search directly into the KAUST repository interface; requires infrastructure planning for concurrent users and managed embeddings
- **Multilingual support** — automatic query translation for Arabic-language queries; deferred until Arabic embedding models mature

## License

MIT
