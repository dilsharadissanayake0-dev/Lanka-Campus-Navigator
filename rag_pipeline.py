import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

def initialize_rag_corpus():
    os.makedirs("data", exist_ok=True)
    pdf_path = "data/ugc_handbook.pdf"
    txt_path = "data/ugc_handbook_corpus.txt"
    
    documents = []
    
    # 1. Load PDF if available in data/
    if os.path.exists(pdf_path):
        print(f"[INFO] Found {pdf_path}. Loading PDF handbook...")
        pdf_loader = PyPDFLoader(pdf_path)
        documents.extend(pdf_loader.load())
    
    # 2. Load Fallback Text Corpus
    if os.path.exists(txt_path):
        print(f"[INFO] Loading text corpus from {txt_path}...")
        txt_loader = TextLoader(txt_path, encoding="utf-8")
        documents.extend(txt_loader.load())

    if not documents:
        print("[WARNING] No documents found to ingest!")
        return None

    # Chunking Strategy for RAG
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = text_splitter.split_documents(documents)

    # Embeddings & Vector Database Persistence
    embeddings = FastEmbedEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"[SUCCESS] ChromaDB initialized with {len(chunks)} document chunks!")
    return vector_store

if __name__ == "__main__":
    initialize_rag_corpus()
