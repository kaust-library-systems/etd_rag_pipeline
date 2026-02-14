# Test querying the database.
# Marcelo Garcia

from pathlib import Path
from langchain_ollama import OllamaEmbeddings
import chromadb


def main():
    vector_store_db = Path("/data") / "ETD_rag" / "etd_rag.db"

    client = chromadb.PersistentClient(str(vector_store_db))

    collection = client.get_collection("ETD")

    print(f"The collection '{collection.name}' has {collection.count()} records.")

    embed_model = "granite-embedding:30m"
    embeddings = OllamaEmbeddings(model=embed_model)

    query_text = (
        "Which of the documents are about non-disjunction events during meiosis"
    )
    query_embeddings = embeddings.embed_query(query_text)

    results = collection.query(
        query_embeddings=query_embeddings,
        include=["metadatas"],
        n_results=5,
    )

    metadatas = results["metadatas"]

    for mm in metadatas:
        print(mm[0]["source"])

    print("Have a nice day!")


if __name__ == "__main__":
    main()
