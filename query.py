# Query the ETD vector store (etd_pillars collection).
# Marcelo Garcia

import csv
from configparser import ConfigParser
from os import PathLike
from pathlib import Path

import requests
import typer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

REPOSITORY_API_URL = "https://repository.kaust.edu.sa/server/api/"
HANDLE_URL = "https://hdl.handle.net/"


def get_item_metadata(handle: str) -> dict:
    """Fetch public metadata for a DSpace item by handle.

    Args:
        handle: bare DSpace handle, e.g. '10754/625510'

    Returns:
        Parsed item metadata as a dictionary.
    """
    response = requests.get(
        f"{REPOSITORY_API_URL}pid/find",
        params={"id": handle},
        headers={"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()

    return response.json()


def get_abstract(handle: str) -> str:
    """Fetch the abstract for a DSpace item by bare handle, e.g. '10754/133211'."""
    try:
        item = get_item_metadata(handle)
        return item["metadata"]["dc.description.abstract"][0]["value"].replace("\n", " ")
    except Exception:
        return "Abstract not available."


def dedupe_by_source(results: list[Document]) -> list[Document]:
    """Keep the best-ranked chunk per source document, preserving rank order."""
    seen: dict[str, Document] = {}
    for result in results:
        source = result.metadata.get("source")
        if source not in seen:
            seen[source] = result
    return list(seen.values())


def save_results_csv(
    results: list[dict],
    output_file: str | PathLike[str] = "results.csv",
) -> None:
    fieldnames = [
        "handle",
        "url",
        "title",
        "author",
        "type",
        "year",
        "advisor",
        "department",
        "section",
        "abstract",
    ]
    output_path = Path(output_file)

    with open(output_path, mode="w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def search(query: str, embeddings, vector_store, config: ConfigParser) -> list[dict]:
    query_embeddings = embeddings.embed_query(query)
    kk = int(config["query"]["k"])
    fetch_k = int(config["query"]["fetch_k"])

    results = vector_store.max_marginal_relevance_search_by_vector(
        query_embeddings, k=kk, fetch_k=fetch_k
    )

    items = []
    for chunk in dedupe_by_source(results):
        meta = chunk.metadata
        handle = meta.get("handle", "")
        items.append(
            {
                "handle": handle,
                "url": f"{HANDLE_URL}{handle}" if handle else "",
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "type": meta.get("type", ""),
                "year": meta.get("year", ""),
                "advisor": meta.get("advisor", ""),
                "department": meta.get("department", ""),
                "section": meta.get("section") or "",
                "abstract": get_abstract(handle),
            }
        )

    return items


def main(query: str):
    config = ConfigParser()
    config.read(Path(__file__).parent / "config.ini")

    embeddings = OllamaEmbeddings(model=config["embeddings"]["model"])

    vector_store = Chroma(
        collection_name=config["chromadb"]["collection"],
        embedding_function=embeddings,
        host=config["chromadb"]["host"],
        port=int(config["chromadb"]["port"]),
    )

    results = search(query, embeddings, vector_store, config)
    output = f"{config['paths']['output_dir']}/results.csv"
    save_results_csv(results, output)
    print(f"{len(results)} documents found. Have a nice day!")


if __name__ == "__main__":
    typer.run(main)
