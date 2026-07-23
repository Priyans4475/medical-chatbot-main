import streamlit as st

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


@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return db


# @st.cache_resource
# def load_llm():
#     return ChatOllama(
#         model="llama3.2",      # Change if you downloaded another model
#         temperature=0,
#     )

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
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


def main():

    st.set_page_config(
        page_title="Medical Chatbot",
        page_icon="🩺",
    )

    st.title("🩺 Medical Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a medical question...")

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        try:

            db = load_vectorstore()

            retriever = db.as_retriever(
                search_kwargs={"k": 3}
            )

            llm = load_llm()

            document_chain = create_stuff_documents_chain(
                llm,
                prompt,
            )

            retrieval_chain = create_retrieval_chain(
                retriever,
                document_chain,
            )

            response = retrieval_chain.invoke(
                {
                    "input": question
                }
            )

            answer = response["answer"]

            with st.chat_message("assistant"):
                st.markdown(answer)

            with st.expander("📚 Source Documents"):

                for i, doc in enumerate(response["context"], 1):

                    source = doc.metadata.get("source", "Unknown File")
                    page = doc.metadata.get("page", "Unknown")

                    st.markdown(f"### 📄 Source {i}")

                    st.write(f"**File:** {source}")
                    st.write(f"**Page:** {page + 1 if isinstance(page, int) else page}")

                    st.write("**Preview:**")
                    st.info(doc.page_content[:500] + "...")

                    st.markdown("---")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

        except Exception as e:

            st.error(str(e))


if __name__ == "__main__":
    main()