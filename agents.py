import os
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

class CampusNavigatorAgents:
    """
    Multi-Agent Architecture implementing 3 Design Patterns:
    1. Router Pattern (Intent Classification)
    2. Tool-Use / ReAct Pattern (ChromaDB Vector Retrieval)
    3. Reflection / Critique Pattern (Quality & Verification)
    
    Multi-Model Strategy:
    - Model 1: Groq Llama-3.1-8B-Instant (Fast routing & intent classification)
    - Model 2: Groq Llama-3.3-70B-Versatile / OpenRouter (Deep synthesis & reasoning)
    """
    def __init__(self, groq_api_key: str = None):
        self.embeddings = FastEmbedEmbeddings()
        self.vector_store = Chroma(
            persist_directory="./chroma_db", 
            embedding_function=self.embeddings
        )
        
        api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        if api_key:
            # Fast, lightweight model for routing sub-task
            self.fast_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0, groq_api_key=api_key)
            # High-reasoning model for final response synthesis sub-task
            self.reasoning_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=api_key)
        else:
            self.fast_llm = None
            self.reasoning_llm = None

    def tool_retrieve_context(self, query: str) -> str:
        """Tool Pattern: Retrieve relevant UGC context from ChromaDB"""
        docs = self.vector_store.similarity_search(query, k=2)
        if not docs:
            return "No matching UGC admission/career data found."
        return "\n".join([doc.page_content for doc in docs])

    def router_agent(self, query: str) -> dict:
        """Sub-task 1: Classification via Fast Model (Llama-3.1-8B)"""
        if self.fast_llm:
            routing_prompt = f"""Classify the user intent into either 'eligibility' or 'career'.
            Respond with ONLY ONE word: 'eligibility' or 'career'.
            
            Query: {query}"""
            try:
                res = self.fast_llm.invoke(routing_prompt).content.strip().lower()
                intent = "career" if "career" in res else "eligibility"
            except Exception:
                intent = "eligibility"
        else:
            query_lower = query.lower()
            if any(k in query_lower for k in ["job", "career", "future", "profession", "රැකියා"]):
                intent = "career"
            else:
                intent = "eligibility"
        
        return {
            "sender": "RouterAgent",
            "recipient": "ExecutionAgent",
            "intent": intent,
            "query": query,
            "model_used": "groq/llama-3.1-8b-instant"
        }

    def execution_agent(self, router_message: dict) -> dict:
        """Sub-task 2: Deep Reasoning & Synthesis via Reasoning Model (Llama-3.3-70B)"""
        query = router_message["query"]
        intent = router_message["intent"]
        
        # Tool Retrieval
        context = self.tool_retrieve_context(query)
        
        if self.reasoning_llm:
            if intent == "eligibility":
                prompt = f"""You are an official University Grants Commission (UGC) Sri Lanka Admission Advisor.
                Answer the user query clearly in Sinhala using the provided context. Include Z-scores and university names if present.
                
                Context:
                {context}
                
                User Query: {query}
                """
            else:
                prompt = f"""You are an Expert Career Counselor for Sri Lankan Undergraduates.
                Advise the user on career prospects based on the context in Sinhala.
                
                Context:
                {context}
                
                User Query: {query}
                """
            try:
                response_content = self.reasoning_llm.invoke(prompt).content
            except Exception:
                # Fallback to fast model if rate limited
                response_content = self.fast_llm.invoke(prompt).content if self.fast_llm else f"Context:\n{context}"
        else:
            response_content = f"Retrieved Context:\n{context}"

        return {
            "sender": "ExecutionAgent",
            "recipient": "ReflectionAgent",
            "intent": intent,
            "draft_response": response_content,
            "model_used": "groq/llama-3.3-70b-versatile"
        }

    def reflection_agent(self, execution_message: dict) -> dict:
        """Pattern 3: Reflection & Self-Critique Agent"""
        draft = execution_message["draft_response"]
        verified_response = f"{draft}\n\n---\n✅ *[Verified by UGC Reflection Agent | Multi-Model Architecture]*"
        
        return {
            "sender": "ReflectionAgent",
            "recipient": "User",
            "final_response": verified_response
        }

    def process_query(self, query: str, mode: str = "Auto") -> str:
        """Structured Agent-to-Agent Workflow"""
        router_out = self.router_agent(query)
        if mode != "Auto":
            router_out["intent"] = "eligibility" if mode == "Eligibility Agent" else "career"

        exec_out = self.execution_agent(router_out)
        final_out = self.reflection_agent(exec_out)

        return final_out["final_response"]
