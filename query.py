# Test querying the database.
# Marcelo Garcia

import typer
import csv
import requests
from pathlib import Path
from os import PathLike
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils import get_handler, get_file_metadata
from configparser import ConfigParser

REPOSITORY_API_URL = "https://repository.kaust.edu.sa/server/api/"

def get_files(results: list[Document]) -> list[str]:
    """Return different files in the results from query in the vector store"""

    sources = [result.metadata['source'] for result in results]
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

    fieldnames = ["source", "author", "title", "abstract"]
    output_path = Path(output_file)

    with open(output_path, mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main(query: str):

    config = ConfigParser()
    config.read(Path(__file__).parent / "config.ini")

    #vector_store_db = Path("/data") / "ETD_rag" / "md_db" / "etd_rag.db"

    embed_model = config["embeddings"]["model"]
    embeddings = OllamaEmbeddings(model=embed_model)
    
    # Initialize parameters from config file.
    collection = config["chromadb"]["collection"]
    host = config["chromadb"]["host"]
    port = config["chromadb"]["port"]

    # Initialize the vector store.
    vector_store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        host=host,
        port=port,
    )

    query_embeddings = embeddings.embed_query(query)

    # Using large `k` and `fetch_k` so we can deduplicate the entries. There 
    # are many chunck per document.
    # Using MMR 
    kk = int(config["query"]["k"])
    fetch_k = int(config["query"]["fetch_k"])

    results = vector_store.max_marginal_relevance_search_by_vector(query_embeddings, k=kk, fetch_k=fetch_k)

    # Deduplication.
    print(f"Number of results: {len(results)}")
    sources = get_files(results)

    # Set (read) metadata file
    metadata_path = config["paths"]["metadata"]

    # Initialize path to results file.
    output = f'{config["paths"]["output_dir"]}/results.csv'

    rows = []
    for source in sources:
        source_path = Path(source)
        #print(f"mg: results metadata source: {results[rr].metadata['source']}")
        #print(f"mg: results page content: {results[rr].page_content[:30]}")

        source_info = get_source_info(source_path.name, metadata_path)
        handler = get_handler(source_info['Handle'])
        item = get_item_metadata(handler)
        abstract_raw = item["metadata"]["dc.description.abstract"][0]["value"]
        abstract = abstract_raw.replace('\n',' ')
        #print(f"{source_path}, {source_info['Author']}, {source_info['Title']}")
        print(f"Saving {source_path} to csv file")
        rows.append({
            "source": str(source_path),
            "author": source_info['Author'],
            "title": source_info['Title'],
            "abstract": abstract,
        })

    save_results_csv(rows, output)
    print("Have a nice day!")


if __name__ == "__main__":
    typer.run(main)
