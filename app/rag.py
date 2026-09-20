from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from vectorstore import load_vectore_store

load_dotenv()

api = os.getenv("OPENAI_API_KEY")

# load vectorstore
vector_store = load_vectore_store()

# load retriver
retriver = vector_store.as_retriever(
    search_kwargs={
        "k":3
    }
)

# define LLM

client = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# PromptTemplate
prompt = ChatPromptTemplate.from_template(
    """
    ### Role
    You are a college assisstant chatbot.

    ### Context
    There are details about college present here

    ### Task
    You need to give clear answers for the user query from the documents.

    ### Rules
    - Don't create any new data by yourself.
    - Never give any generic responses.
    - If there's no data present, just tell the user I have no knowledge about it.
    - Never move or deviate from the context.

    Context:
    {context}

    Question:
    {question}

    Answer: 

    """
)

def ask_question(question : str) -> str:

    documents = retriver.invoke(question)

    # combining all retrived content into a single string
    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    # prompt invoke
    messages = prompt.invoke({
        "context":context,
        "question":question
    })

    # final LLM invoke(core)
    response = client.invoke(messages)

    return response.content, documents

# Driver
# ------------------------------------------------

if __name__ == "__main__":

    question = "What attendance is required for semester exams?"
    response,documents = ask_question(question)

    for _,doc in enumerate(documents, start=1):
        print(f"Content: {doc.page_content}")
        print(f"File: {doc.metadata.get("source")}")
        print(f"File Type: {doc.metadata.get("file_type")}")

