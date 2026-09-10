# Experimental chat interface over the ETD vector store (etd_pillars collection).
# Marcelo Garcia

from configparser import ConfigParser
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

CHAT_MODEL = "granite3.1-dense:8b"


def main():
    config = ConfigParser()
    config.read(Path(__file__).parent / "config.ini")

    embeddings = OllamaEmbeddings(model=config["embeddings"]["model"])

    vector_store = Chroma(
        collection_name=config["chromadb"]["collection"],
        embedding_function=embeddings,
        host=config["chromadb"]["host"],
        port=int(config["chromadb"]["port"]),
    )

    question_user = "Which documents investigate first-order mean-field games?"

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question_user)

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the following context: {context}

    Question: {question}
    """
    )

    llm = ChatOllama(
        model=CHAT_MODEL,
        temperature=0,
        validate_model_on_init=True,
    )

    chain = prompt | llm
    answer = chain.invoke({"context": docs, "question": question_user})

    print(answer.content)


if __name__ == "__main__":
    main()
