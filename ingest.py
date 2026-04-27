# Ingest documents (most PDF) and save to a vector store to enable
# semantic search on the collection.
# Marcelo Garcia (marcelo.garcia@kaust.edu.sa)
#

import re
import logging
from pathlib import Path
from os import PathLike
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_classic.indexes import SQLRecordManager, index
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from utils import get_file_metadata
from configparser import ConfigParser

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

def is_arabic_heavy(text: str, threshold: float = 0.3) -> bool:
    """Return True if more than `threshold` fraction of characters are Arabic."""
    arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
    total_chars = len(text.strip())
    if total_chars == 0:
        return False
    return (arabic_chars / total_chars) > threshold

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting ETD RAG ingestion pipeline")
    
    # Read the config file
    config = ConfigParser()
    config.read(Path(__file__).parent / "config.ini")

    #input_path = Path("/data") / "ETD_rag" / "markdown"
    #vector_store_db = Path("/data") / "ETD_rag" / "md_db" / "etd_rag.db"
    #record_manager_db = Path("/data") / "ETD_rag" / "md_db" / "record_manager.db"

    input_path = Path(config["paths"]["input_dir"])
    metadata_path = Path(config["paths"]["metadata"])
    record_manager_db = config["paths"]["record_manager"]
    embed_model = config["embeddings"]["model"]

    logger.info("Input directory: %s", input_path)
    logger.info("Record manager: %s", record_manager_db)

    logger.info("Embeddings model: %s", embed_model)
    embeddings = OllamaEmbeddings(model=embed_model)

    # Initialize the vector store.
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
    logger.info("Vector store host: %s", host)
    logger.info("Vector store port: %s", port)

    #vector_store = Chroma(
    #    collection_name="ETD",
    #    embedding_function=embeddings,
    #    persist_directory=str(vector_store_db),
    #)

    # Initialize the record manager.
    record_manager = SQLRecordManager(
        namespace=config["chromadb"]["collection"],
        db_url=f"sqlite:///{record_manager_db}",
    )
    record_manager.create_schema()

    # List of files to process.
    input_file_list = list_files(input_path)
    logger.info("Found %d files to process", len(input_file_list))

    for file in input_file_list:
        logger.info("Processing file: %s", file.name)

        loader = TextLoader(file, encoding='utf-8')
        docs = loader.load()
    
        # First split by headers to respect document structure
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on = [
                ('#', 'h1'),
                ('##', 'section'),
                ('###', 'subsection'),
            ],
            strip_headers=False,
        )
        header_chunks = md_splitter.split_text(docs[0].page_content)

        # Then split oversized sections (e.g. long chapters) by character
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
        )
        chunks = char_splitter.split_documents(header_chunks)
        logger.info("Chunks before Arabic filter: %d", len(chunks))
        chunks = [c for c in chunks if not is_arabic_heavy(c.page_content)]
        logger.info("Chunks after Arabic filter: %d", len(chunks))

        # Keep only essential metadata to ensure consistent hashing
        # Also, adding metadata from the file to the chunk.
        source_info = get_file_metadata(file.stem + ".pdf", metadata_path) 
        for chunk in chunks:
            chunk.metadata = {
                "source": str(file),
                "section": chunk.metadata.get("section"),
                "title": source_info.get("Title") if source_info else None,
                "author": source_info.get("Author") if source_info else None,
                "handle": source_info.get("Handle") if source_info else None,
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
