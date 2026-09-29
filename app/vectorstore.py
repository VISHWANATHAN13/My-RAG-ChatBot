from pathlib import Path
from langchain_chroma import Chroma
from ingest import load_all_documents, split_documents
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
api = os.getenv("OPENAI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"


# Embedding model initialization
def get_embedding_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api
    )

# create chroma DB
def create_vector_store(chunks):
    embeddings = get_embedding_model()

    vectore_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="college_knowledge"
    )
    return vectore_store

# loading the vector DB (existing)
def load_vectore_store():
    embeddings = get_embedding_model()

    vectore_store = Chroma(
        collection_name="college_knowledge",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    return vectore_store

# Similarity search score
def chroma_retrive(vector_store, query, k=5):
    results = vector_store.similarity_search_with_score(
        query,
        k=k # Top K
    )
    formatted_results = []

    for doc, score in results:
        formatted_results.append({
            "id": doc.metadata.get("id"),
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        })

    return formatted_results



# Driver
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    documents = load_all_documents(str(DATA_DIR))
    chunks = split_documents(documents)
    # vectore_store = create_vector_store(chunks)
    vectore_store = load_vectore_store()

    print(f"Loaded documents: {len(documents)}")
    print(f"Generated documents: {len(chunks)}")

    if vectore_store:
        print("vector database created and chunks are embedded")

    query = "What attendance is required for semester exams?"
    results = chroma_retrive(vectore_store, query, k=3)

    print("=" * 80)
    print(f"Query: {query}")

    for i, result in enumerate(results, start=1):
        print(f"Index: {i}")
        print(f"Metadata: {result['metadata']}")
        print(f"Score: {result['score']}")
        print(f"Content: {result['content']}")
    print("=" * 80)



  
