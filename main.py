from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_core.prompts import PromptTemplate
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_docling.loader import ExportType

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from transformers import AutoTokenizer

FILE_PATH = ["/data/ETD_rag/test/133211.pdf"]
#EMBED_MODEL_ID = "granite-embedding"
EMBED_MODEL_ID = "ibm-granite/granite-embedding-30m-english"
GEN_MODEL_ID = "phi4:14b"
EXPORT_TYPE = ExportType.DOC_CHUNKS
MILVUS_URI = Path("/") / "data" / "ETD_rag"/ "docling.db"

MAX_TOKENS = 64

def main():
    print("Hello from etd-rag-pipeline!")

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
        max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer` for HF case
    )

    loader = DoclingLoader(
        file_path=FILE_PATH,
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=tokenizer)
    )

    docs = loader.load()

    if EXPORT_TYPE == ExportType.DOC_CHUNKS:
        splits = docs
    elif EXPORT_TYPE == ExportType.MARKDOWN:
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
    else:
        raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")

    # Inspecting the split.
    for d in splits[:3]:
        print(f"- {d.page_content=}")
        print("...")

    # Ingestion
    embedding = HuggingFaceEmbeddings(
        model=EMBED_MODEL_ID,
    )

    vectorstore = Milvus.from_documents(
        documents=splits,
        embedding=embedding,
        collection_name="docling_demo",
        connection_args={"uri": str(MILVUS_URI)},
        index_params={"index_type": "FLAT"},
        drop_old=True,
    )

if __name__ == "__main__":
    main()
    print("Have a nice day.")
