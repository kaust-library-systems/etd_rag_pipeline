import csv
import io
import requests
import streamlit as st
from pathlib import Path
from configparser import ConfigParser
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from utils import get_handler, get_file_metadata

REPOSITORY_API_URL = "https://repository.kaust.edu.sa/server/api/"

EXAMPLES = [
    "Numerical methods for fluid dynamics simulation",
    "Deep learning for medical image analysis",
    "Renewable energy storage and optimization",
    "Protein structure prediction and drug discovery",
    "Seawater desalination membrane technology",
]

config = ConfigParser()
config.read(Path(__file__).parent / "config.ini")


@st.cache_resource
def load_resources():
    embeddings = OllamaEmbeddings(model=config["embeddings"]["model"])
    vector_store = Chroma(
        collection_name=config["chromadb"]["collection"],
        embedding_function=embeddings,
        host=config["chromadb"]["host"],
        port=int(config["chromadb"]["port"]),
    )
    return embeddings, vector_store


def fetch_abstract(handle_url: str) -> str:
    try:
        handler = get_handler(handle_url)
        response = requests.get(
            f"{REPOSITORY_API_URL}pid/find",
            params={"id": handler},
            headers={"Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        item = response.json()
        abstract_raw = item["metadata"]["dc.description.abstract"][0]["value"]
        return abstract_raw.replace("\n", " ")
    except Exception:
        return "Abstract not available."


@st.cache_data
def run_search(query: str) -> list[dict]:
    embeddings, vector_store = load_resources()

    k = int(config["query"]["k"])
    fetch_k = int(config["query"]["fetch_k"])
    metadata_path = config["paths"]["metadata"]

    query_embedding = embeddings.embed_query(query)
    results = vector_store.max_marginal_relevance_search_by_vector(
        query_embedding, k=k, fetch_k=fetch_k
    )

    # Deduplicate by source, keep first matching chunk per document
    seen = {}
    for doc in results:
        source = doc.metadata["source"]
        if source not in seen:
            seen[source] = doc

    items = []
    for source, doc in seen.items():
        source_path = Path(source)
        info = get_file_metadata(source_path.stem + ".pdf", metadata_path)
        if not info:
            continue

        handle_url = info.get("Handle", "")
        abstract = fetch_abstract(handle_url)

        items.append({
            "handle": handle_url,
            "title": info.get("Title", ""),
            "author": info.get("Author", ""),
            "type": info.get("Type", ""),
            "section": doc.metadata.get("section", ""),
            "abstract": abstract,
        })

    return items


def results_to_csv(results: list[dict]) -> str:
    output = io.StringIO()
    fieldnames = ["title", "author", "type", "handle", "section", "abstract"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    return output.getvalue()


# --- UI ---

st.set_page_config(page_title="ETD Search", layout="centered")
st.title("ETD Search")
st.caption("Search theses and dissertations from the KAUST repository.")

query_input = st.text_input(
    "query",
    placeholder="Search for theses and dissertations...",
    label_visibility="collapsed",
)

selected_example = st.pills(
    "Examples",
    options=EXAMPLES,
    selection_mode="single",
    label_visibility="collapsed",
)

query = query_input or selected_example

if query:
    with st.spinner("Searching..."):
        results = run_search(query)

    st.write(f"**{len(results)} documents found**")

    if results:
        csv_data = results_to_csv(results)
        st.download_button(
            label="Export as CSV",
            data=csv_data,
            file_name="etd_results.csv",
            mime="text/csv",
        )

        st.divider()

        for item in results:
            with st.container():
                st.markdown(f"#### [{item['title']}]({item['handle']})")
                col1, col2 = st.columns(2)
                col1.markdown(f"**Author:** {item['author']}")
                col2.markdown(f"**Type:** {item['type']}")
                if item["section"]:
                    st.markdown(f"**Matching section:** {item['section']}")
                with st.expander("Abstract"):
                    st.write(item["abstract"])
                st.divider()
