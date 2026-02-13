import streamlit as st
import sys
from pathlib import Path
import platform
import os
import json
import uuid

# Check if running in the correct virtual environment
if sys.prefix == sys.base_prefix:
    st.error("This app is not running in the correct virtual environment. Please run the setup script first.")
    st.code("cd Vivekananda-rag/streamlit_app && bash setup.sh")
    st.stop()

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# --- Model and Backend Setup ---
if platform.system() == "Darwin":
    from mlx_rag.mlx_rag_system import VivekanandaRAGSystem
    backend = "MLX (Apple Silicon)"
else:
    from ollama_rag.ollama_rag_system import VivekanandaRAGSystem
    backend = "Ollama"

# --- Page Config ---
st.set_page_config(layout="wide")

# --- RAG System Loading ---
@st.cache_resource
def load_rag_system(model_name):
    rag = VivekanandaRAGSystem(model_name=model_name)
    rag.load_index()
    return rag

# --- UI ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("Swami Vivekananda RAG System")
    model_option = st.selectbox("Choose a model:", ("Fast", "Deep"), key="model_selection", label_visibility="collapsed")
    st.caption(f"Running on: {backend}")

with col3:
    if st.button("Export Chat as JSON"):
        chat_json = json.dumps(st.session_state.messages, indent=4)
        st.download_button(
            label="Download JSON",
            data=chat_json,
            file_name="chat_history.json",
            mime="application/json",
        )


model_name = "bge-small-v1.5" if "Fast" in model_option else "e5-large-v2"
rag_system = load_rag_system(model_name)


# --- Chat History Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# --- Function to handle new prompts ---
def handle_prompt(prompt):
    st.session_state.processing = True
    st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})

    with st.spinner("Searching for an answer..."):
        response = rag_system.answer(prompt)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response['answer'],
            "sources": response['sources'],
            "followup_questions": response.get('followup_questions', []),
            "id": str(uuid.uuid4())
        })
    st.session_state.processing = False

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**Source {i+1} (Similarity: {source['similarity']:.4f})**")
                        st.write(source['text'])
                        st.write(f"**Metadata:** {source['metadata']}")
            
            if "followup_questions" in message and message["followup_questions"]:
                st.subheader("Suggested Questions:")
                for i, q in enumerate(message["followup_questions"]):
                    if st.button(q, key=f"followup_{message['id']}_{i}", disabled=st.session_state.processing):
                        handle_prompt(q)
                        st.rerun()

# --- Handle new prompts ---
if prompt := st.chat_input("What is your question?"):
    handle_prompt(prompt)
    st.rerun()

# --- CSS for styling ---
st.markdown("""
<style>
    .stSelectbox {
        width: 150px !important;
    }
</style>
""", unsafe_allow_html=True)