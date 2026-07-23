import tempfile
from pathlib import Path

import streamlit as st

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_FAISS_PATH = "vectorstore/uploaded_docs_faiss"
SUPPORTED_TYPES = ["pdf", "txt", "md"]


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():
    return ChatOllama(
        model="tinyllama",
        temperature=0,
    )


prompt = ChatPromptTemplate.from_template(
    """
You are a helpful medical assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
reply exactly:

"I don't know."

Context:
{context}

Question:
{input}
"""
)


def load_documents_from_uploads(uploaded_files):
    documents = []
    temp_dir = Path(tempfile.mkdtemp(prefix="uploaded_docs_", dir="."))

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_path = temp_dir / file_name

        with file_path.open("wb") as f:
            f.write(uploaded_file.getbuffer())

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif suffix in {".txt", ".md"}:
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                st.warning(f"Unsupported file type skipped: {file_name}")
                continue

            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source"] = file_name

            documents.extend(loaded_docs)

        except Exception as exc:
            st.warning(f"Could not read {file_name}: {exc}")

    return documents


def build_vectorstore(documents):
    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    embeddings = load_embeddings()
    db = FAISS.from_documents(chunks, embeddings)

    Path(DB_FAISS_PATH).parent.mkdir(parents=True, exist_ok=True)
    db.save_local(DB_FAISS_PATH)

    return db


def load_existing_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def main():
    st.set_page_config(
        page_title="Upload Your Own Documents Chatbot",
        page_icon="📄",
    )

    st.title("📄 Upload Documents Chatbot")
    st.caption("Upload one or more PDF/TXT/MD files and chat with the combined content.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []

    if "vectorstore_ready" not in st.session_state:
        st.session_state.vectorstore_ready = False

    with st.sidebar:
        st.header("Upload documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=SUPPORTED_TYPES,
        )

        if st.button("Embed uploaded documents") and uploaded_files:
            new_documents = load_documents_from_uploads(uploaded_files)
            if new_documents:
                st.session_state.uploaded_documents.extend(new_documents)
                st.session_state.vectorstore = build_vectorstore(
                    st.session_state.uploaded_documents
                )
                st.session_state.vectorstore_ready = True
                st.success(
                    f"Embedded {len(st.session_state.uploaded_documents)} document(s) together."
                )
            else:
                st.warning("No readable documents were found.")

        if st.button("Clear knowledge base"):
            st.session_state.messages = []
            st.session_state.uploaded_documents = []
            st.session_state.vectorstore_ready = False
            if "vectorstore" in st.session_state:
                del st.session_state.vectorstore

        if st.session_state.uploaded_documents:
            st.subheader("Loaded files")
            for doc in st.session_state.uploaded_documents:
                source = doc.metadata.get("source", "unknown")
                st.write(f"- {source}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.vectorstore_ready:
        st.info("Upload one or more documents, then click 'Embed uploaded documents' to build the knowledge base.")

    question = st.chat_input("Ask a question about your uploaded documents...")

    if question:
        if not st.session_state.vectorstore_ready:
            st.warning("Please build the knowledge base first.")
            return

        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        try:
            db = st.session_state.vectorstore
            retriever = db.as_retriever(search_kwargs={"k": 3})
            llm = load_llm()

            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            response = retrieval_chain.invoke({"input": question})
            answer = response["answer"]

            with st.chat_message("assistant"):
                st.markdown(answer)

                with st.expander("Source Documents"):
                    for i, doc in enumerate(response["context"], 1):
                        st.markdown(f"### Document {i}")
                        st.write(doc.page_content)

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
