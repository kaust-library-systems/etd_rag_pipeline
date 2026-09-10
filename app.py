import csv
import io
from configparser import ConfigParser
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from query import search

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


@st.cache_data
def run_search(query: str) -> list[dict]:
    embeddings, vector_store = load_resources()

    return search(query, embeddings, vector_store, config)


def results_to_csv(results: list[dict]) -> str:
    output = io.StringIO()
    fieldnames = [
        "handle",
        "url",
        "title",
        "author",
        "type",
        "year",
        "advisor",
        "department",
        "section",
        "abstract",
    ]
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
                st.markdown(f"#### [{item['title']}]({item['url']})")
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"**Author:** {item['author']}")
                col2.markdown(f"**Type:** {item['type']}")
                col3.markdown(f"**Year:** {item['year']}")
                if item["department"]:
                    st.markdown(f"**Department:** {item['department']}")
                if item["section"]:
                    st.markdown(f"**Matching section:** {item['section']}")
                with st.expander("Abstract"):
                    st.write(item["abstract"])
                st.divider()
