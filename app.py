import streamlit as st
from src.graph import build_research_graph

# Page Configuration
st.set_page_config(
    page_title="Kestrel Research Assistant",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Kestrel Research Assistant")
st.caption("Multi-Agent Research & Compliance Pipeline powered by LangGraph, ChromaDB, and Groq")

# Initialize Chat Memory in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Load compiled state graph (cached to prevent re-instantiation on rerun)
@st.cache_resource
def load_graph():
    return build_research_graph()

graph_app = load_graph()

# Sidebar Control Panel
st.sidebar.title("Controls")
if st.sidebar.button("Clear Chat Memory", use_container_width=True):
    st.session_state.messages = []
    st.session_state.conversation_history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Agent Pipeline Graph")
st.sidebar.markdown("""
1. **Planner**: Resolves context & rewrites query.
2. **Retriever**: Queries ChromaDB vector store.
3. **Verifier**: Audits chunk relevance & evidence.
4. **Synthesizer**: Generates cited answer.
""")

# Render Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "agent_status" in msg and msg["agent_status"]:
            with st.expander("Show Agent Execution Steps"):
                for step in msg["agent_status"]:
                    st.write(f"- {step}")

# User Input Box
if prompt := st.chat_input("Ask a question about Kestrel documentation..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process Query through Multi-Agent State Graph
    with st.chat_message("assistant"):
        with st.status("Executing Multi-Agent Graph...", expanded=True) as status:
            initial_state = {
                "question": prompt,
                "conversation_history": st.session_state.conversation_history,
                "rewritten_query": "",
                "retrieved_chunks": [],
                "verifier_verdict": "",
                "verifier_reasoning": "",
                "final_answer": "",
                "citations": [],
                "agent_status": []
            }
            
            # Execute LangGraph pipeline
            result = graph_app.invoke(initial_state)
            
            # Render intermediate step indicators
            for step in result.get("agent_status", []):
                st.write(f"✓ {step}")
            
            verdict = result.get("verifier_verdict", "unknown")
            status.update(
                label=f"Execution Complete | Audit Verdict: {verdict.upper()}", 
                state="complete", 
                expanded=False
            )

        final_answer = result.get("final_answer", "")
        st.markdown(final_answer)

        # Store Assistant Output
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "agent_status": result.get("agent_status", [])
        })
        
        # Track Multi-turn History
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        st.session_state.conversation_history.append({"role": "assistant", "content": final_answer})