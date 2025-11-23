from langchain_ollama import ChatOllama

def main():
    print("Hello from etd-rag-pipeline!")

    llm = ChatOllama(
        model="granite4:3b",
        temperature=0,
    )

    message = [
        (
            "system",
            "You are a helpful assistant that translates English to Spanish. Translate the user sentence.",
        ),
        (
            "human",
            "I love programming."
        ),
    ]


    ai_msg = llm.invoke(message)
    print(ai_msg.content)


if __name__ == "__main__":
    main()
