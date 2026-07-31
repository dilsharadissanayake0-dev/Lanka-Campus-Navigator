# 🎓 Lanka Campus Navigator

**Lanka Campus Navigator** is an Agentic AI application designed for Sri Lankan G.C.E. A/L students and undergraduates. It offers intelligent UGC admission eligibility checks, Z-score analysis, and career guidance using a Multi-Agent architecture backed by Retrieval-Augmented Generation (RAG).

---

## 🏗️ Architecture & Design Patterns

The application demonstrates **3 core Agentic AI Design Patterns**:

1. **Router Pattern**: Classifies incoming queries into `eligibility` or `career` intents to route tasks efficiently.
2. **Tool-Use / ReAct Pattern**: Retrieves verified domain knowledge from a **ChromaDB Vector Database** containing Sri Lankan UGC guidelines.
3. **Reflection / Self-Critique Pattern**: Analyzes draft outputs before presenting the final response to ensure factual correctness and adherence to UGC guidelines.

---

## 🧠 Multi-Model Selection Strategy

| Sub-task / Role | Model Selected | Provider | Justification |
| :--- | :--- | :--- | :--- |
| **Intent Routing & Classification** | `llama-3.1-8b-instant` | Groq | Ultra-low latency (<200ms), zero cost, high accuracy. |
| **Deep Reasoning & Response Synthesis** | `llama-3.3-70b-versatile` | Groq | Superior context understanding, Sinhala language generation. |

---

## 🚀 Local Setup & Installation

```bash
pip install -r requirements.txt
python rag_pipeline.py
streamlit run app.py
```
