# Test querying the database.
# Marcelo Garcia

import csv
from pathlib import Path
from os import PathLike
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


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


def main():
    vector_store_db = Path("/data") / "ETD_rag" / "etd_rag.db"

    embed_model = "granite-embedding:30m"
    embeddings = OllamaEmbeddings(model=embed_model)

    # Initialize the vector store.
    vector_store = Chroma(
        collection_name="ETD",
        embedding_function=embeddings,
        persist_directory=str(vector_store_db),
    )

    query_text = "Which document investigates first-order mean-field game?"
    query_embeddings = embeddings.embed_query(query_text)

    results = vector_store.similarity_search_by_vector(query_embeddings)

    source = results[0].metadata['source']

    source_path = Path(source)
    #print(f"mg: results metadata source: {results[rr].metadata['source']}")
    #print(f"mg: results page content: {results[rr].page_content[:30]}")

    source_info = get_source_info(source_path.name)
    print(f"{source_path}, {source_info['Author']}, {source_info['Title']}")
    print("Have a nice day!")


if __name__ == "__main__":
    main()
