from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, AIMessage, SystemMessage

def main():
    print("Hello from etd-rag-pipeline!")

    model = ChatOllama(
        model="granite4:3b",
        temperature=0,
    )

    system_msg = SystemMessage(
        '''You are a helpful assistent that respond to questions with three
           exclamation marks.'''
    )

    human_msg = HumanMessage("What is the capital of France?")

    answer = model.invoke([system_msg, human_msg])
    print(answer.content)


if __name__ == "__main__":
    main()
    print("Have a nice day.")
