# Test querying the database.
# Marcelo Garcia

import csv
import typer
import requests
from pathlib import Path
from os import PathLike
from urllib.parse import urlparse
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

REPOSITORY_API_URL = "https://repository.kaust.edu.sa/server/api/"

def get_handler(handle_url: str) -> str:
    """
    Return the just the handler from the handle url
    Input: http://hdl.handle.net/10754/224071 
    Return: 10754/224071
    """

    return urlparse(handle_url).path[1:]

def get_files(results: list[Document]) -> list[str]:
    """Return different files in the results from query in the vector store"""

    sources = [result.metadata['source'] for result in results]
    return list(dict.fromkeys(sources))

def get_file_metadata(
    filename: str,
    metadata_file: str | PathLike[str] = "metadata.csv",
) -> dict | None:
    """Look up metadata for a file in the CSV.

    Args:
        filename: The filename to search for in the 'File' column.
        metadata_file: Path to the CSV file.

    Returns:
        Dictionary with the file's metadata, or None if not found.
    """
    metadata_path = Path(metadata_file)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_path, encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if row["File"] == filename:
                return dict(row)

    return None


def get_source_info(filename: str) -> dict | None:
    """Look up source information for a file in the CSV.
    Args: filename: The filename to search for in the 'File' column.
    """

    metadata_path = Path("/data") / "ETD_rag" / "metadata.csv"
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


def main(query: str, k: int = 50, fetch_k: int = 170):
    vector_store_db = Path("/data") / "ETD_rag" / "etd_rag.db"

    embed_model = "granite-embedding:30m"
    embeddings = OllamaEmbeddings(model=embed_model)

    # Initialize the vector store.
    vector_store = Chroma(
        collection_name="ETD",
        embedding_function=embeddings,
        persist_directory=str(vector_store_db),
    )

    query_embeddings = embeddings.embed_query(query)

    # Using large `k` and `fetch_k` so we can deduplicate the entries. There 
    # are many chunck per document.
    # Using MMR 
    results = vector_store.max_marginal_relevance_search_by_vector(query_embeddings, k=k, fetch_k=fetch_k)

    # Deduplication.
    print(f"Number of results: {len(results)}")

    sources = get_files(results)
    seen = set()
    unique_results = []
    for result in results:
        source = result.metadata['source']
        if source not in seen:
            seen.add(source)
            unique_results.append(result)

    for source in sources:
        source_path = Path(source)
        #print(f"mg: results metadata source: {results[rr].metadata['source']}")
        #print(f"mg: results page content: {results[rr].page_content[:30]}")

        source_info = get_source_info(source_path.name)
        handler = get_handler(source_info['Handle'])
        item = get_item_metadata(handler)
        abstract = item["metadata"]["dc.description.abstract"][0]["value"]
        print(f"{source_path}, {source_info['Author']}, {source_info['Title']}, {abstract}")
    print("Have a nice day!")


if __name__ == "__main__":
    typer.run(main)
