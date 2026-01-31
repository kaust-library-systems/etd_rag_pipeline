# Ingest documents (most PDF) and save to a vector store to enable
# semantic search on the collection.
# Marcelo Garcia (marcelo.garcia@kaust.edu.sa)
# 

from pathlib import Path
from os import PathLike
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def list_files(input_dir: str | PathLike[str]) -> list[Path]:
    """Return a list of files as Path objects in a directory.

    The 'input_dir' can be passed as string or Path object."""

    # Basic sanity checks for the input directory:
    input_resolve = Path(input_dir).resolve()

    if not input_resolve.exists():
        raise ValueError(f"Input directory '{input_dir}' does not exists")
    if not input_resolve.is_dir():
        raise ValueError(f"Supplied input directory '{input_dir}' it's not a directory")

    return [item for item in input_resolve.iterdir() if item.is_file()]


def main():
    print("Hello from convert2md")

    input_path = Path("/data") / "ETD_rag" / "test"
    vector_store_db = Path("/data") / "ETD_rag" / "etd_rag.db"

    embeddings = OllamaEmbeddings(model="granite4:3b")

    # Initialize the vector store.
    vector_store = Chroma(
        collection_name="ETD",
        embedding_function=embeddings,
        persist_directory=str(vector_store_db),
    )

    # List of files to process.
    input_file_list = list_files(input_path)

    for file in input_file_list:
        print(f"Processing file {file}")

        loader = PyPDFLoader(file)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        chunks = splitter.split_documents(docs)

        print(f"Splitted the document into '{len(chunks)}' chunks")
        print("Saving document to vector store.")
        chunks_id = [str(uuid4()) for _ in range(len(chunks))]
        vector_store.add_documents(documents=chunks, ids=chunks_id)


if __name__ == "__main__":
    main()
