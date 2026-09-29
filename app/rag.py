from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from collections import defaultdict

from vectorstore import load_vectore_store
from bm25_retrival import format_docs, bm25_retrive

load_dotenv()

api = os.getenv("OPENAI_API_KEY")

# load vectorstore
vector_store = load_vectore_store()

# BM25 index is built from the same chunks stored in Chroma, so both
# retrievers see exactly the same corpus
stored_data = vector_store.get()
bm25, bm25_documents = format_docs(stored_data)

# retriver
# -------------------------------------------------------------------
# Fusion key: the stored chunks carry no chunk_id, and Chroma's
# similarity_search does not return its internal ids, so the chunk text
# is the only identifier shared by both retrievers.
def doc_key(doc):
    return doc["content"]

# semantic retrival
def semantic_retrive(query, k=3):
    results = vector_store.similarity_search(
        query,
        k=k
    )
    # return retriver.invoke(query)
    formatted_results = []
    for doc in results:
        formatted_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    return formatted_results


# Bm25 retrival is imported from bm25_retrival.py

# -------------------------------------------------------------------

# RRF scoring
# -------------------------------------------------------------------
# RRF Results
def reciprocal_rank_fusion(result_lists,k=60):
    """
    result_lists:
    [
        dense_results,
        bm25_results
    ]
    """
    fused_scores = defaultdict(float)
    docs = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            doc_id = doc_key(doc)
            docs[doc_id] = doc
            fused_scores[doc_id] += 1 / (k + rank + 1)

    reranked = sorted(
        fused_scores.items(),
        key = lambda x : x[1],
        reverse= True
    )

    return [
        {**docs[doc_id], "score": score}
        for doc_id, score in reranked
        ]
# -------------------------------------------------------------------

# Driver method to call and merge both scores
# ------------------------------------------------------------------
def hybrid_search(query, bm25, stored_data, top_k = 5):

    dense_results = semantic_retrive(query, k=3)
    bm25_results = bm25_retrive(query, bm25, stored_data, k=20)

    fused_results = reciprocal_rank_fusion([
        dense_results,
        bm25_results
    ])

    return fused_results[:top_k]
# ------------------------------------------------------------------


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

    documents = hybrid_search(question, bm25, stored_data)

    # combining all retrived content into a single string
    context = "\n\n".join(
        doc["content"]
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

    print(f"Question: {question}")
    print(f"Answer: {response}")

    print("\n" + "=" * 80)
    print("Retrieved context (hybrid / RRF)")
    print("=" * 80)

    for rank,doc in enumerate(documents, start=1):
        print(f"\nRank: {rank}")
        print(f"RRF Score: {doc['score']}")
        print(f"File: {doc['metadata'].get("source")}")
        print(f"File Type: {doc['metadata'].get("file_type")}")
        print(f"Content: {doc['content']}")

