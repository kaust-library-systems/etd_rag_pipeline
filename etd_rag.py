# Ingest documents (most PDF) and save to a vector store to enable
# semantic search on the collection.
# Marcelo Garcia (marcelo.garcia@kaust.edu.sa)
#

import logging
from pathlib import Path
from os import PathLike
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.indexes import SQLRecordManager, index
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


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
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting ETD RAG ingestion pipeline")

    input_path = Path("/data") / "ETD_rag" / "test"
    vector_store_db = Path("/data") / "ETD_rag" / "etd_rag.db"
    record_manager_db = Path("/data") / "ETD_rag" / "record_manager.db"

    logger.info("Input directory: %s", input_path)
    logger.info("Vector store: %s", vector_store_db)
    logger.info("Record manager: %s", record_manager_db)

    embed_model = "granite-embedding:30m"
    logger.info("Embeddings model: %s", embed_model)
    embeddings = OllamaEmbeddings(model=embed_model)

    # Initialize the vector store.
    vector_store = Chroma(
        collection_name="ETD",
        embedding_function=embeddings,
        persist_directory=str(vector_store_db),
    )

    # Initialize the record manager.
    record_manager = SQLRecordManager(
        namespace="ETD",
        db_url=f"sqlite:///{record_manager_db}",
    )
    record_manager.create_schema()

    # List of files to process.
    input_file_list = list_files(input_path)
    logger.info("Found %d files to process", len(input_file_list))

    for file in input_file_list:
        logger.info("Processing file: %s", file.name)

        loader = PyPDFLoader(file)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        chunks = splitter.split_documents(docs)

        # Keep only essential metadata to ensure consistent hashing
        for chunk in chunks:
            chunk.metadata = {
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
            }

        logger.info("Split document into %d chunks", len(chunks))

        # Debug: show metadata of first chunk
        if chunks:
            logger.debug("First chunk metadata: %s", chunks[0].metadata)
            logger.debug("First chunk content (100 chars): %s", chunks[0].page_content[:100])

        logger.info("Indexing document to vector store")
        result = index(
            chunks,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
            key_encoder="blake2b",
        )
        logger.info(
            "Indexing complete: added=%d, skipped=%d, deleted=%d",
            result["num_added"],
            result["num_skipped"],
            result["num_deleted"],
        )

    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()
