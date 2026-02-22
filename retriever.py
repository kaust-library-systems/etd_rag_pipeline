# Test querying the database.
# Marcelo Garcia

import csv
from pathlib import Path
from os import PathLike
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import chromadb


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

    client = chromadb.PersistentClient(path=str(vector_store_db))
    collection = client.get_collection("ETD")
    
    question_user = "Which documents discuss  investigates first-order mean-field game"

    vector_store = Chroma(
        client=client,
        collection_name="ETD",
        embedding_function=embeddings,
    )

    retriever = vector_store.as_retriever()
    docs = retriever.invoke(question_user)

    llm = ChatOllama(
        model="granite3.1-dense:8b",
        temperature=0,
        validate_model_on_init=True,
    )

    prompt = ChatPromptTemplate.from_template("""Answer the question based on the following context: {context}

    Question: {question}
    """)

    chain = prompt | llm
    answer = chain.invoke(
        {"context": docs, "question": question_user}
    )

    print(f"The answer was {answer}")


if __name__ == "__main__":
    main()
