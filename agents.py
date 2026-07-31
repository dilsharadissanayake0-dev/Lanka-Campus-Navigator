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
            self.router_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0, groq_api_key=api_key)
            self.agent_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=api_key)
        else:
            self.router_llm = None
            self.agent_llm = None

    def retrieve_context(self, query: str) -> str:
        docs = self.vector_store.similarity_search(query, k=3)
        if not docs:
            return "No matching UGC guidelines found."
        return "\n".join([doc.page_content for doc in docs])

    def eligibility_agent(self, z_score: str, stream: str, district: str, user_query: str) -> str:
        """Agent 1: Dedicated UGC Eligibility & Campus Admission Agent"""
        context = self.retrieve_context(f"{stream} {district} {user_query}")
        
        prompt = f"""You are the official UGC Admission & Eligibility Agent.
        Provide a clear, direct, and encouraging response in simple English based on the student's details and UGC context.

        Student Profile:
        - Z-Score: {z_score}
        - Stream: {stream}
        - District: {district}
        - Question: {user_query}

        Context:
        {context}

        Provide details on:
        1. Campus/Course Eligibility based on their Z-Score and Stream.
        2. Any required University Aptitude Tests (e.g., Moratuwa Architecture, Kelaniya Computing, Translation Studies).
        """
        
        if self.agent_llm:
            return self.agent_llm.invoke(prompt).content
        return f"Retrieved UGC Context:\n{context}"

    def career_guidance_agent(self, stream: str, degree_interest: str, user_query: str) -> str:
        """Agent 2: Dedicated Career & Job Guidance Agent"""
        context = self.retrieve_context(f"{stream} {degree_interest} {user_query}")
        
        prompt = f"""You are an Expert Sri Lankan Undergraduate Career Guidance Counselor.
        Provide actionable career guidance in simple English.

        Student Profile:
        - Stream: {stream}
        - Degree/Interest: {degree_interest}
        - Question: {user_query}

        Context:
        {context}

        Provide details on:
        1. Top Career Pathways & Job Roles available in Sri Lanka & globally.
        2. Key Skills required to succeed in this industry.
        """
        
        if self.agent_llm:
            return self.agent_llm.invoke(prompt).content
        return f"Retrieved Career Context:\n{context}"
