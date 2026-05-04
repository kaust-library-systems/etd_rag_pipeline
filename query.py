# Test querying the database.
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

from utils import get_file_metadata, get_handler

REPOSITORY_API_URL = "https://repository.kaust.edu.sa/server/api/"


def get_files(results: list[Document]) -> list[str]:
    """Return different files in the results from query in the vector store"""

    sources = [result.metadata["source"] for result in results]
    return list(dict.fromkeys(sources))


def get_source_info(filename: str, metadata_path: str) -> dict | None:
    """Look up source information for a file in the CSV.
    Args: filename: The filename to search for in the 'File' column.
    """

    info = get_file_metadata(filename, metadata_path)

    return info


def get_item_metadata(handle: str) -> dict:
    """
    Fetch public metadata for a DSpace item by handle.
    Args:
    handle: DSpace handle, e.g. '10754/625510'

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


def save_results_csv(
    results: list[dict],
    output_file: str | PathLike[str] = "results.csv",
) -> None:
    fieldnames = ["handle", "title", "author", "type", "section", "abstract"]
    output_path = Path(output_file)

    with open(output_path, mode="w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def search(query: str, embeddings, vector_store, config: ConfigParser) -> list[dict]:
    query_embeddings = embeddings.embed_query(query)
    kk = int(config["query"]["k"])
    fetch_k = int(config["query"]["fetch_k"])
    metadata_path = config["paths"]["metadata"]

    results = vector_store.max_marginal_relevance_search_by_vector(
        query_embeddings, k=kk, fetch_k=fetch_k
    )

    sources = get_files(results)

    items = []
    for source in sources:
        source_path = Path(source)
        source_info = get_source_info(source_path.name, metadata_path)
        if not source_info:
            continue
        handler = get_handler(source_info["Handle"])
        item = get_item_metadata(handler)
        abstract_raw = item["metadata"]["dc.description.abstract"][0]["value"]
        abstract = abstract_raw.replace("\n", " ")
        matching_chunk = next(
            (r for r in results if r.metadata["source"] == source), None
        )
        section = matching_chunk.metadata.get("section", "") if matching_chunk else ""
        items.append(
            {
                "handle": source_info["Handle"],
                "title": source_info["Title"],
                "author": source_info["Author"],
                "type": source_info["Type"],
                "section": section,
                "abstract": abstract,
            }
        )

    return items


def main(query: str):
    config = ConfigParser()
    config.read(Path(__file__).parent / "config.ini")

    embed_model = config["embeddings"]["model"]
    embeddings = OllamaEmbeddings(model=embed_model)

    # Initialize parameters from config file.
    collection = config["chromadb"]["collection"]
    host = config["chromadb"]["host"]
    port = int(config["chromadb"]["port"])

    # Initialize the vector store.
    vector_store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        host=host,
        port=port,
    )

    results = search(query, embeddings, vector_store, config)
    output = f"{config['paths']['output_dir']}/results.csv"
    save_results_csv(results, output)
    print("Have a nice day!")


if __name__ == "__main__":
    typer.run(main)
