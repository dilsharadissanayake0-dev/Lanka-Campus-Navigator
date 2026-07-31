import streamlit as st
import os
from agents import CampusNavigatorAgents

st.set_page_config(
    page_title="Lanka Campus Navigator", 
    page_icon="🎓", 
    layout="wide"
)

st.title("🎓 Lanka Campus Navigator")
st.subheader("UGC Admission, Z-Score Eligibility & Career Guidance Multi-Agent System")

groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

@st.cache_resource
def load_navigator(api_key: str):
    return CampusNavigatorAgents(groq_api_key=api_key)

navigator = load_navigator(groq_api_key)

st.sidebar.header("⚙️ System Configuration")

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")

agent_mode = st.sidebar.radio(
    "Select Agent Execution Mode:",
    ["Auto Router", "Eligibility Agent", "Career Guidance Agent"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**System Architecture:**\n"
    "- **Router Agent:** Groq Llama-3.1-8B (Fast classification)\n"
    "- **Execution Agent:** Groq Llama-3.3-70B (Deep reasoning & RAG retrieval)\n"
    "- **Reflection Agent:** Quality assurance & verification\n"
    "- **Vector Store:** ChromaDB with FastEmbed"
)

user_query = st.text_input(
    "Enter your question regarding UGC Cut-off marks, Universities, or Career options:",
    placeholder="e.g., කැලණිය Software Engineering එකට ඕන Z-Score එක කීයද?"
)

if st.button("Submit Query", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Processing query via Multi-Agent AI System..."):
            mode_mapping = {
                "Auto Router": "Auto",
                "Eligibility Agent": "Eligibility Agent",
                "Career Guidance Agent": "Career Guidance Agent"
            }
            selected_mode = mode_mapping[agent_mode]
            try:
                response = navigator.process_query(user_query, mode=selected_mode)
                st.markdown("### 🤖 Agentic Response:")
                st.success(response)
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")

st.markdown("---")
st.caption("Powered by RAG Architecture, ChromaDB, Groq Multi-Models & Multi-Agent Orchestration.")
