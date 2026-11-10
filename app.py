import csv
import io
from configparser import ConfigParser
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from query import search

# Per-collection presentation. Keys are the labels in config.ini [collections].
SOURCES = {
    "Theses": {
        "noun": "theses and dissertations",
        "examples": [
            "Numerical methods for fluid dynamics simulation",
            "Deep learning for medical image analysis",
            "Renewable energy storage and optimization",
            "Protein structure prediction and drug discovery",
            "Seawater desalination membrane technology",
        ],
        "csv_fields": [
            "handle", "url", "title", "author", "type", "year",
            "advisor", "department", "section", "abstract",
        ],
        "csv_name": "etd_results.csv",
    },
    "Articles": {
        "noun": "research articles",
        "examples": [
            "Coral reef bleaching in the Red Sea",
            "Perovskite solar cell stability",
            "Machine learning for protein design",
            "Forward osmosis membranes for desalination",
            "Salt tolerance in crops",
        ],
        "csv_fields": [
            "handle", "url", "title", "author", "type", "year",
            "journal", "doi", "department", "section", "abstract",
        ],
        "csv_name": "article_results.csv",
    },
}

MAX_AUTHORS_SHOWN = 3

config = ConfigParser()
config.optionxform = str  # keep the label case in [collections]
config.read(Path(__file__).parent / "config.ini")
COLLECTIONS = dict(config["collections"])


@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model=config["embeddings"]["model"])


@st.cache_resource
def load_vector_store(collection: str):
    return Chroma(
        collection_name=collection,
        embedding_function=load_embeddings(),
        host=config["chromadb"]["host"],
        port=int(config["chromadb"]["port"]),
    )


@st.cache_data
def run_search(query: str, collection: str) -> list[dict]:
    return search(query, load_embeddings(), load_vector_store(collection), config)


def results_to_csv(results: list[dict], fieldnames: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    return output.getvalue()


def short_authors(authors: str) -> str:
    """Articles can have dozens of authors; show the first few."""
    names = [a.strip() for a in authors.split(";") if a.strip()]
    if len(names) <= MAX_AUTHORS_SHOWN:
        return "; ".join(names)
    return "; ".join(names[:MAX_AUTHORS_SHOWN]) + " et al."


# --- UI ---

st.set_page_config(page_title="KAUST Repository Search", layout="centered")
st.title("KAUST Repository Search")

label = st.radio(
    "Collection",
    options=list(COLLECTIONS),
    horizontal=True,
    label_visibility="collapsed",
)
source = SOURCES[label]
collection = COLLECTIONS[label]

st.caption(f"Search {source['noun']} from the KAUST repository.")

query_input = st.text_input(
    "query",
    placeholder=f"Search for {source['noun']}...",
    label_visibility="collapsed",
)

selected_example = st.pills(
    "Examples",
    options=source["examples"],
    selection_mode="single",
    label_visibility="collapsed",
    key=f"examples_{label}",
)

query = query_input or selected_example

if query:
    with st.spinner("Searching..."):
        results = run_search(query, collection)

    st.write(f"**{len(results)} documents found**")

    if results:
        st.download_button(
            label="Export as CSV",
            data=results_to_csv(results, source["csv_fields"]),
            file_name=source["csv_name"],
            mime="text/csv",
        )

        st.divider()

        for item in results:
            with st.container():
                st.markdown(f"#### [{item['title']}]({item['url']})")
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"**Author:** {short_authors(item['author'])}")
                col2.markdown(f"**Type:** {item['type']}")
                col3.markdown(f"**Year:** {item['year']}")
                if item["advisor"]:
                    st.markdown(f"**Advisor:** {item['advisor']}")
                if item["journal"]:
                    st.markdown(f"**Journal:** {item['journal']}")
                if item["doi"]:
                    st.markdown(f"**DOI:** [{item['doi']}](https://doi.org/{item['doi']})")
                if item["department"]:
                    st.markdown(f"**Department:** {item['department']}")
                if item["section"]:
                    st.markdown(f"**Matching section:** {item['section']}")
                with st.expander("Abstract"):
                    st.write(item["abstract"])
                st.divider()
