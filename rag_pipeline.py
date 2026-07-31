import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

def initialize_rag_corpus():
    os.makedirs("data", exist_ok=True)
    
    # Detailed English Dataset for UGC Admission & Career Pathways
    ugc_data = """
    OFFICIAL UGC SRI LANKA ADMISSION & CAREER GUIDELINE CORPUS

    1. ENGINEERING STREAM (Cut-offs & Eligibility):
    - Moratuwa Engineering: Z-score ~ 1.95 (Colombo/Gampaha), 1.88 (Kandy/Galle). Requires Aptitude Test for Architecture/Design.
    - Peradeniya Engineering: Z-score ~ 1.85.
    - Ruhuna Engineering: Z-score ~ 1.72.
    - Aptitude Tests: Moratuwa Architecture, Fashion Design, IT & Management require university-level Aptitude Tests.

    2. PHYSICAL SCIENCE & COMPUTING STREAM:
    - University of Colombo (Computer Science - BCS): Z-score ~ 1.55. Requires Computing Aptitude Test.
    - University of Kelaniya (Software Engineering): Z-score ~ 1.45. Requires Software Engineering Aptitude Test.
    - University of Moratuwa (Information Technology - IT): Z-score ~ 1.60.
    - Career Pathways: Software Engineer, Data Scientist, Full-Stack Developer, AI Engineer, Cybersecurity Analyst.

    3. ARTS & HUMANITIES STREAM:
    - University of Colombo (Arts/International Relations): Z-score ~ 1.40 (Kandy/Colombo).
    - University of Sri Jayewardenepura (Humanities/Social Sciences): Z-score ~ 1.30.
    - University of Peradeniya (Arts/Psychology/Economics): Z-score ~ 1.25.
    - Aptitude Tests: Translation Studies (Kelaniya/Jaffna), Mass Communication (Kelaniya), Visual Arts/Music require practical aptitude evaluations.
    - Career Pathways: Civil Services Officer, Lecturer/Academic, Economist, Psychologist, Diplomat, Content Strategist, Public Relations Specialist.

    4. COMMERCE & MANAGEMENT STREAM:
    - Sri Jayewardenepura Management: Z-score ~ 1.65.
    - Colombo Business Administration: Z-score ~ 1.60.
    - Career Pathways: Financial Analyst, Business Analyst, Marketing Manager, HR Executive, Auditor.
    """

    corpus_path = "data/ugc_handbook_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(ugc_data)

    loader = TextLoader(corpus_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embeddings = FastEmbedEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("[SUCCESS] ChromaDB successfully re-initialized with English Corpus!")
    return vector_store

if __name__ == "__main__":
    initialize_rag_corpus()
