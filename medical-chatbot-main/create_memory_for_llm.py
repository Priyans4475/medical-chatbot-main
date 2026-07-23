from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"


def load_pdf_files():
    """
    Load all PDF files from the data folder.
    """
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    return loader.load()


def create_chunks(documents):
    """
    Split documents into smaller chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    return splitter.split_documents(documents)


def get_embedding_model():
    """
    Load HuggingFace embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_db():

    print("Loading PDF files...")
    documents = load_pdf_files()
    print(f"Loaded {len(documents)} pages.")

    print("Creating chunks...")
    chunks = create_chunks(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model...")
    embeddings = get_embedding_model()

    print("Creating FAISS vector store...")
    db = FAISS.from_documents(chunks, embeddings)

    Path(DB_FAISS_PATH).parent.mkdir(parents=True, exist_ok=True)

    db.save_local(DB_FAISS_PATH)

    print("\n✅ Vector database created successfully!")
    print(f"Saved to: {DB_FAISS_PATH}")


if __name__ == "__main__":
    create_vector_db()