import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

def initialize_rag_corpus():
    os.makedirs("data", exist_ok=True)
    
    # Domain-Specific UGC Handbook Knowledge Base Corpus (Sinhala & English)
    ugc_data = """
    University Grants Commission (UGC) Sri Lanka - Admission & Cut-off Guidelines

    1. Engineering Stream:
    - University of Moratuwa (Engineering): Minimum Z-Score 1.95 (Colombo/Gampaha).
    - University of Peradeniya (Engineering): Minimum Z-Score 1.85.
    - University of Ruhuna (Engineering): Minimum Z-Score 1.72.
    - University of Jaffna (Engineering): Minimum Z-Score 1.68.
    - Career Pathways: Software Engineer, Civil Engineer, Electronics Specialist, Mechatronics Engineer.

    2. Biological Science & Medicine Stream:
    - Faculty of Medicine (University of Colombo): Minimum Z-Score 2.10.
    - Faculty of Medicine (University of Sri Jayewardenepura): Minimum Z-Score 2.02.
    - Faculty of Medicine (University of Peradeniya): Minimum Z-Score 1.98.
    - Faculty of Medicine (University of Ruhuna): Minimum Z-Score 1.90.
    - Career Pathways: Medical Officer, Biomedical Scientist, Healthcare Administrator, Pharmacologist.

    3. Physical Science & Computing Stream:
    - University of Colombo (Computer Science - BCS): Minimum Z-Score 1.55.
    - University of Kelaniya (Software Engineering): Minimum Z-Score 1.45.
    - University of Sri Jayewardenepura (Applied Sciences): Minimum Z-Score 1.30.
    - University of Moratuwa (Information Technology - IT): Minimum Z-Score 1.60.
    - Career Pathways: Data Scientist, Full-Stack Developer, AI Researcher, Cybersecurity Specialist.

    4. Management & Commerce Stream:
    - University of Sri Jayewardenepura (Management/Finance): Minimum Z-Score 1.65.
    - University of Colombo (Business Administration): Minimum Z-Score 1.60.
    - University of Kelaniya (Commerce): Minimum Z-Score 1.40.
    - Career Pathways: Business Analyst, Financial Analyst, Marketing Executive, HR Manager.
    """

    corpus_path = "data/ugc_handbook_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(ugc_data)

    # Document Loading & Chunking Strategy
    loader = TextLoader(corpus_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # FastEmbed Embeddings & ChromaDB Vector Store Persistence
    embeddings = FastEmbedEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("[SUCCESS] ChromaDB Vector Store successfully initialized!")
    return vector_store

if __name__ == "__main__":
    initialize_rag_corpus()
