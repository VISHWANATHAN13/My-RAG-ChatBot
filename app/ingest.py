from langchain_community.document_loaders import TextLoader, CSVLoader, BSHTMLLoader
from langchain_core.documents import Document
from pathlib import Path
from typing import List
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Loader part starts
# ------------------------------------------------------------------------
# Text Loader

def load_text_documents(data_dir: str) -> List[Document]:

    documents = []

    for file_path in Path(data_dir).rglob("*.txt"):
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

        docs = loader.load()

        for doc in docs:
            doc.metadata.update({
                "source": file_path.name,
                "file_path": str(file_path),
                "file_type": "txt"
            })

        documents.extend(docs)

    return documents

# debug
# documents = load_text_documents("./data")
# print(f"Loaded {len(documents)} documents")


#  CSV loader
def load_csv_documents(data_dir : str) -> List[Document]:

    documents = []

    for file_path in Path(data_dir).rglob("*.csv"):
        loader = CSVLoader(
            file_path=str(file_path),
            encoding="utf-8"
        )

        docs = loader.load()

        for doc in docs:
            doc.metadata.update({
                "source":file_path.name,
                "file_path":str(file_path),
                "file_type":"csv"
            })
        documents.extend(docs)
    return documents

# Load JSON
def load_json_documents(data_dir : str) -> list[Document]:

    documents = []

    for file_path in Path(data_dir).rglob("*.json"):
        with open(file_path,"r",encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "faqs" in data:
            for faq in data["faqs"]:
                content = (
                    f"Category: {faq.get('category','')}\n"
                    f"Question: {faq.get('question')}\n"
                    f"Answer: {faq.get('answer')}\n"
                    f"Keywords: {', '.join(faq.get('keywords',[]))}"
                ) 

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": file_path.name,
                            "file_path": str(file_path),
                            "file_type": "json",
                            "faq_id" : faq.get("id","")

                        }
                    )
                )
    return documents

# Load HTML
def load_html_documents(data_dir : str) -> list[Document]:

    documents = []

    for file_path in Path(data_dir).rglob("*.html"):
        loader = BSHTMLLoader(
            str(file_path),
            open_encoding="utf-8"
        )

        docs = loader.load()

        for doc in docs:
            doc.metadata.update({
                "source": file_path.name,
                "file_path": str(file_path),
                "file_type": "html"
            })
        documents.extend(docs)

    return documents

# Driver method to call all Loaders
def load_all_documents(data_dir : str) -> list[Document]:

    documents = []

    documents.extend(load_text_documents(data_dir))
    documents.extend(load_csv_documents(data_dir))
    documents.extend(load_html_documents(data_dir))
    documents.extend(load_json_documents(data_dir))

    return documents

# Loader part ends
# -------------------------------------------------------------------------

# Splitter
# -------------------------------------------------------------------------

def split_documents(documents : list[Document]) -> list[Document]:

    # Using RecursiveCharacterTextSplitter for optimal approach
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True
    )

    chunks = splitter.split_documents(documents)

    return chunks

# Chunking ends
# -------------------------------------------------------------------------

#  Invoke method
# -------------------------------------------------------------------------
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"

    documents = load_all_documents(str(data_dir))
    chunked_documents = split_documents(documents)

    print(len(documents))
    print(f"Loaded {len(documents)} documents\n")
    print(f"Chunked {len(chunked_documents)} documents\n")

    print("Loader section")
    print("="*80)
    for i, doc in enumerate(documents, start=1):
        print(f"Document {i}")
        print(f"Metadata:", doc.metadata)
        print("Content:", doc.page_content[:300])
        print("-" * 50)
    print("="*80)

    print("Chunking section")
    print("="*50)
    for i, doc in enumerate(chunked_documents,start=1):
        print(f"Chunk {i}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content}")
        print("-" * 50)
    print("="*80)




# invoke ends
# -------------------------------------------------------------------------


    
        