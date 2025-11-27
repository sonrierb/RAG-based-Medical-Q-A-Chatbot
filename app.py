import streamlit as st
import os
import uuid
import tempfile
from model import MedicalRAGSystem

st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🏥", layout="wide")

def main():
    st.title("🏥 Medical Document RAG Chatbot")
    st.markdown("Upload medical documents and chat with the AI assistant.")

    if "rag_system" not in st.session_state:
        st.session_state.rag_system = MedicalRAGSystem()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "current_doc_id" not in st.session_state:
        st.session_state.current_doc_id = None

    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Document")

        uploaded_file = st.file_uploader("Upload Medical File", 
                                         type=["pdf", "docx", "txt"])

        if uploaded_file is not None:
            doc_id = str(uuid.uuid4())[:8]

            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    try:
                        ext = uploaded_file.name.split(".")[-1].lower()

                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            path = tmp.name

                        text = st.session_state.rag_system.extract_text_from_file(path, ext)

                        if text:
                            docs = st.session_state.rag_system.chunk_text(text)
                            st.session_state.rag_system.create_vector_store(docs, doc_id)

                            st.session_state.document_processed = True
                            st.session_state.current_doc_id = doc_id
                            st.session_state.chat_history = []

                            st.success(f"Document processed! ID: {doc_id}")
                            st.info(f"Chunks created: {len(docs)}")

                        os.unlink(path)

                    except Exception as e:
                        st.error(str(e))

        if st.session_state.current_doc_id:
            st.markdown("---")
            st.subheader("📄 Current Document")
            st.write(f"ID: `{st.session_state.current_doc_id}`")

            if st.button("Clear Document"):
                st.session_state.document_processed = False
                st.session_state.current_doc_id = None
                st.session_state.chat_history = []
                st.session_state.rag_system.vector_store = None
                st.rerun()

    # Chat area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Chat")

        for chat in st.session_state.chat_history:
            role = "You" if chat["type"] == "user" else "Assistant"
            st.markdown(f"**{role}:** {chat['message']}")
            st.markdown("---")

        user_query = st.chat_input("Ask anything about the uploaded medical document...",
                                   disabled=not st.session_state.document_processed)

        if user_query:
            st.session_state.chat_history.append({"type": "user", "message": user_query})

            with st.spinner("Processing..."):
                ans = st.session_state.rag_system.process_query(user_query)

            st.session_state.chat_history.append({"type": "assistant", "message": ans})
            st.rerun()

    with col2:
        st.subheader("ℹ️ Instructions")
        st.write("""
        1. Upload a medical document  
        2. Click **Process Document**  
        3. Ask questions about the content  
        """)

        if st.session_state.chat_history:
            st.subheader("📊 Stats")
            st.write(f"Questions: {len([m for m in st.session_state.chat_history if m['type']=='user'])}")
            st.write(f"Answers: {len([m for m in st.session_state.chat_history if m['type']=='assistant'])}")

if __name__ == "__main__":
    main()
