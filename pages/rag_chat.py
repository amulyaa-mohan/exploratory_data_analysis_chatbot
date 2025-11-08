# pages/rag_chat.py — GENERAL VERSION: Works for ANY PDF (fast, reliable, friendly)
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from src.config.settings import Settings
import tempfile
from pathlib import Path
import torch  # For GPU if available

# ------------------------------------------------------------------
# 1. CACHED EMBEDDINGS (runs only once)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI brain… (first time only)")
def get_embeddings():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": device}
        )
    except:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

# ------------------------------------------------------------------
# 2. LLM
# ------------------------------------------------------------------
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Free tier: 1M tokens/min
        google_api_key=Settings().GOOGLE_API_KEY,
        temperature=0
    )

# ------------------------------------------------------------------
# 3. PROMPT (from file – keep ethical + citation rules)
# ------------------------------------------------------------------
try:
    PROMPT_TPL = Path("src/prompts/rag_prompt.txt").read_text(encoding="utf-8")
except FileNotFoundError:
    st.error("Create `src/prompts/rag_prompt.txt`")
    st.stop()

# ------------------------------------------------------------------
# 4. MAIN APP
# ------------------------------------------------------------------
def run():
    st.title("Document Intelligence Bot")
    st.caption("Ask anything – fast, reliable, and friendly")

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = None

    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "docx"])

    # ------------------------------------------------------------------
    # INDEX DOCUMENT (general – no hard-coded summary)
    # ------------------------------------------------------------------
    if uploaded_file and uploaded_file.name != st.session_state.file_name:
        with st.spinner("Reading document… "):
            tmp_path = Path(tempfile.gettempdir()) / uploaded_file.name
            tmp_path.write_bytes(uploaded_file.getvalue())

            # Loader (general for PDF/TXT/DOCX)
            if uploaded_file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(str(tmp_path))
            elif uploaded_file.name.lower().endswith(".txt"):
                loader = TextLoader(str(tmp_path), encoding="utf-8")
            else:
                loader = Docx2txtLoader(str(tmp_path))

            docs = loader.load()  # Each doc has metadata like 'page'

            # Chunking – 800 chars → fast for any doc
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(docs)

            # Optional: Add a general summary chunk from first page (dynamic)
            if docs:
                first_page = docs[0].page_content[:500].replace("\n", " ").strip()
                summary = Document(
                    page_content=f"Document summary: {first_page}",
                    metadata={"page": "Summary"}
                )
                chunks = [summary] + chunks

            # Build FAISS
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.file_name = uploaded_file.name
            st.session_state.messages = []

            st.success("Ready! Ask anything.")

        tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # QA CHAIN
    # ------------------------------------------------------------------
    if st.session_state.vectorstore:
        qa = RetrievalQA.from_chain_type(
            llm=get_llm(),
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TPL)},
        )

        # Chat
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the document…"):
            # ----- GREETING -----
            if prompt.strip().lower() in ["hello", "hi", "hey"]:
                reply = "Hello! How can I help you with the document?"
            else:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        result = qa.invoke({"query": prompt})
                        reply = result["result"]
                        st.write(reply)

                        # ----- SHOW SOURCES -----
                        if sources := result.get("source_documents"):
                            with st.expander(f"Sources ({len(sources)})"):
                                for doc in sources[:3]:
                                    page = doc.metadata.get("page", "Summary")
                                    snippet = doc.page_content.replace("\n", " ")[:200]
                                    st.caption(f"**Page {page}**: {snippet}…")

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": reply})