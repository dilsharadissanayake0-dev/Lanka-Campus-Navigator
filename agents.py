import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

class CampusNavigatorAgents:
    def __init__(self, groq_api_key: str = None):
        self.embeddings = FastEmbedEmbeddings()
        self.vector_store = Chroma(
            persist_directory="./chroma_db", 
            embedding_function=self.embeddings
        )
        
        api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        if api_key:
            self.fast_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0, groq_api_key=api_key)
            self.agent_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=api_key)
        else:
            self.fast_llm = None
            self.agent_llm = None

    def retrieve_context(self, query: str) -> str:
        docs = self.vector_store.similarity_search(query, k=4)
        if not docs:
            return "No matching UGC guidelines found in database."
        return "\n".join([doc.page_content for doc in docs])

    def eligibility_agent(self, z_score: str, stream: str, district: str, user_query: str = "") -> str:
        """Agent 1: Official UGC Eligibility & Campus Admission Agent"""
        context = self.retrieve_context(f"{stream} stream Z-score {z_score} {district} district {user_query}")
        
        prompt = f"""You are the official UGC Sri Lanka Admission & Eligibility Agent.
        Provide a clear, precise, and encouraging response in simple English based on official UGC guidelines.

        Student Profile:
        - Z-Score: {z_score}
        - Stream: {stream}
        - District: {district}
        - Question: {user_query}

        Context from Knowledge Base:
        {context}

        Tasks:
        1. List eligible Universities and Degree Programs for this Z-score, Stream, and District.
        2. Explicitly specify any required University Aptitude Tests (e.g., Moratuwa Architecture, Kelaniya Computing, Translation Studies).
        """
        if self.agent_llm:
            return self.agent_llm.invoke(prompt).content
        return f"Retrieved UGC Context:\n{context}"

    def career_guidance_agent(self, stream: str, degree_interest: str, user_query: str = "") -> str:
        """Agent 2: Dedicated Career & Job Guidance Agent"""
        context = self.retrieve_context(f"{stream} {degree_interest} career pathways job opportunities Sri Lanka {user_query}")
        
        prompt = f"""You are an Expert Sri Lankan Undergraduate Career Guidance Counselor.
        Provide practical and actionable career advice in simple English.

        Student Profile:
        - Stream: {stream}
        - Degree / Interest: {degree_interest}
        - Question: {user_query}

        Context:
        {context}

        Tasks:
        1. Recommend top 3-4 Career Pathways & Job Roles in Sri Lanka and globally.
        2. Highlight essential Technical and Soft Skills required in the industry.
        """
        if self.agent_llm:
            return self.agent_llm.invoke(prompt).content
        return f"Retrieved Career Context:\n{context}"
