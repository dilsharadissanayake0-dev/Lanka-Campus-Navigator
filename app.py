import streamlit as st
import os
from agents import CampusNavigatorAgents

st.set_page_config(page_title="Lanka Campus Navigator", page_icon="🎓", layout="wide")

st.title("🎓 Lanka Campus Navigator")
st.write("Sri Lankan UGC Campus Admission Eligibility & Career Guidance Multi-Agent System")

groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

@st.cache_resource
def load_navigator(api_key: str):
    return CampusNavigatorAgents(groq_api_key=api_key)

navigator = load_navigator(groq_api_key)

# Clean Tab Architecture for 2 Dedicated Agents
tab1, tab2 = st.tabs(["🎯 Agent 1: Admission & Eligibility Check", "💼 Agent 2: Career Guidance"])

# -------------------------------------------------------------
# TAB 1: AGENT 1 - ELIGIBILITY AGENT
# -------------------------------------------------------------
with tab1:
    st.header("UGC Campus Eligibility & Aptitude Test Checker")
    st.write("Enter your G.C.E. A/L details below to check eligible universities and required aptitude tests.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        z_score = st.text_input("Z-Score:", value="1.80")
    with col2:
        stream = st.selectbox("A/L Stream:", ["Arts", "Physical Science / Maths", "Biological Science", "Commerce", "Technology"])
    with col3:
        district = st.text_input("District:", value="Kandy")

    user_query_1 = st.text_input("Specific Admission Query (Optional):", value="Which campuses can I apply for and are there any aptitude tests?")

    if st.button("Check Eligibility", type="primary"):
        with st.spinner("Agent 1 (Eligibility Agent) is analyzing UGC cutoff databases..."):
            response = navigator.eligibility_agent(z_score, stream, district, user_query_1)
            st.success("### 🤖 Eligibility Agent Response:")
            st.markdown(response)

# -------------------------------------------------------------
# TAB 2: AGENT 2 - CAREER GUIDANCE AGENT
# -------------------------------------------------------------
with tab2:
    st.header("Undergraduate Career Counselor")
    st.write("Explore future career pathways and job prospects based on your academic stream.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        stream_2 = st.selectbox("Your Stream:", ["Arts", "Physical Science / Maths", "Biological Science", "Commerce", "Technology"], key="career_stream")
    with col_b:
        degree_interest = st.text_input("Degree Interest / Subject Area:", value="Psychology / Economics / IT")

    user_query_2 = st.text_input("Career Question:", value="What are the top job roles and career prospects for this degree?")

    if st.button("Get Career Guidance", type="primary"):
        with st.spinner("Agent 2 (Career Agent) is generating career pathway insights..."):
            response = navigator.career_guidance_agent(stream_2, degree_interest, user_query_2)
            st.success("### 💼 Career Guidance Agent Response:")
            st.markdown(response)

st.markdown("---")
st.caption("Lanka Campus Navigator | Powered by Groq LLMs, ChromaDB Vector Store & Multi-Agent Architecture.")
