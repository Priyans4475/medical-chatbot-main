from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

DB_FAISS_PATH = "vectorstore/db_faiss"

from dotenv import load_dotenv
import os

load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI


# ------------------------
# Embeddings
# ------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ------------------------
# Load FAISS
# ------------------------

db = FAISS.load_local(
    DB_FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


# # ------------------------
# # Ollama
# # ------------------------

# llm = ChatOllama(
#     model="llama3.2",
#     temperature=0,
# )

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# ------------------------
# Prompt
# ------------------------

prompt = ChatPromptTemplate.from_template(
    """
Use ONLY the following context to answer.

If the answer is not contained in the context,
reply exactly:

I don't know.

Context:
{context}

Question:
{input}
"""
)


# ------------------------
# Create Chain
# ------------------------

document_chain = create_stuff_documents_chain(
    llm,
    prompt,
)

retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain,
)


# ------------------------
# Ask Questions
# ------------------------

while True:

    question = input("\nAsk Question (exit to quit): ")

    if question.lower() == "exit":
        break

    response = retrieval_chain.invoke(
        {
            "input": question
        }
    )

    print("\nAnswer:\n")
    print(response["answer"])