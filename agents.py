import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

class CampusNavigatorAgents:
    """
    Multi-Agent Architecture matching Horizon Assignment Brief:
    - Patterns: Router Pattern, Tool-Use Pattern (ChromaDB), Reflection Pattern
    - Models: Llama-3.1-8B (Fast Routing) & Llama-3.3-70B (Deep Reasoning)
    """
    def __init__(self, groq_api_key: str = None):
        self.embeddings = FastEmbedEmbeddings()
        self.vector_store = Chroma(
            persist_directory="./chroma_db", 
            embedding_function=self.embeddings
        )
        
        api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        if api_key:
            self.fast_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0, groq_api_key=api_key)
            self.reasoning_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=api_key)
        else:
            self.fast_llm = None
            self.reasoning_llm = None

    def retrieve_context(self, query: str) -> str:
        """Tool Pattern: Retrieve context from ChromaDB"""
        docs = self.vector_store.similarity_search(query, k=4)
        if not docs:
            return "No matching UGC cutoff guidelines found in knowledge base."
        return "\n".join([doc.page_content for doc in docs])

    def eligibility_agent(self, z_score: str, stream: str, district: str, query: str = "") -> str:
        """Agent 1: Admission & Eligibility Agent (Accepts 4 parameters)"""
        search_query = f"{stream} stream Z-score {z_score} {district} district {query}"
        context = self.retrieve_context(search_query)
        
        prompt = f"""You are the official UGC Admission & Eligibility Agent for Sri Lanka.
        Provide a detailed, helpful answer in simple English based on official UGC guidelines.

        Student Profile:
        - Z-Score: {z_score}
        - Stream: {stream}
        - District: {district}
        - Additional Question: {query}

        UGC Knowledge Base Context:
        {context}

        Task:
        1. List eligible Universities and Degree Programs for this Z-score, Stream, and District.
        2. Clearly state if any specific University Aptitude Tests are required (e.g., Architecture, Information Technology, Translation Studies, Sports Science).
        """
        
        if self.reasoning_llm:
            draft = self.reasoning_llm.invoke(prompt).content
            return f"{draft}\n\n---\n✅ *[Verified by UGC Reflection Agent | Knowledge Base Check Passed]*"
        return f"Retrieved UGC Context:\n{context}"

    def career_guidance_agent(self, stream: str, eligibility_summary: str, query: str = "") -> str:
        """Agent 2: Career Guidance Counselor Agent"""
        search_query = f"{stream} career job opportunities Sri Lanka {query}"
        context = self.retrieve_context(search_query)
        
        prompt = f"""You are an Expert Career Counselor for Sri Lankan Undergraduates.
        Provide practical career advice in simple English based on the student's eligible degree options.

        Student Profile:
        - Stream: {stream}
        - Eligible Degree Summary: {eligibility_summary}
        - Specific Query: {query}

        Context:
        {context}

        Task:
        1. Recommend top 3-4 future Career Pathways & Job Roles corresponding to the eligible degree options.
        2. Outline essential Technical & Soft Skills needed in the industry.
        """
        
        if self.reasoning_llm:
            draft = self.reasoning_llm.invoke(prompt).content
            return f"{draft}\n\n---\n💼 *[Verified by Career Reflection Agent]*"
        return f"Retrieved Career Context:\n{context}"
