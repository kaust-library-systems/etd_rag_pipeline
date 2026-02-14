# Test querying the database.
# Marcelo Garcia

from pathlib import Path
from langchain_chroma import Chroma

def main():
    vector_store_db = Path("/data") / "ETD_rag" / "etd_rag.db"


if __name__ == "__main__":
    main()