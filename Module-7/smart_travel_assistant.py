import hashlib
import io
import os
from typing import List

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


#
# Configure the app and load local environment settings
#
# Load values such as OPENAI_API_KEY from the local .env file
load_dotenv()

# Page settings are defined before other Streamlit elements
st.set_page_config(
    page_title="PDF Question-Answering Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF Question-Answering Chatbot")
st.caption(
    "Upload a PDF, ask questions, and get answers grounded only in the document."
)


#
# Helper functions for processing PDFs and building the QA pipeline
#
def get_pdf_hash(pdf_bytes: bytes) -> str:
    """Return a stable hash so the same uploaded PDF is not re-embedded."""
    return hashlib.sha256(pdf_bytes).hexdigest()


def extract_pdf_pages(pdf_bytes: bytes) -> List[Document]:
    """
    Extract text from every PDF page and preserve page-number metadata.
    Each page becomes one LangChain Document before chunking.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    documents: List[Document] = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": page_index + 1},
                )
            )

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split pages into overlapping chunks.

    chunk_size=1000 gives embeddings enough local context.
    chunk_overlap=200 helps preserve information crossing chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk"] = index

    return chunks


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """Generate OpenAI embeddings and store them in a FAISS vector database."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


def format_docs(documents: List[Document]) -> str:
    """Convert retrieved documents into a context string for the model."""
    formatted = []

    for doc in documents:
        page = doc.metadata.get("page", "unknown")
        chunk = doc.metadata.get("chunk", "unknown")
        formatted.append(
            f"[Page {page}, Chunk {chunk}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted)


def create_qa_chain(retriever):
    """
    Build the LangChain question-answering pipeline.

    Required runnable components:
    - RunnableParallel
    - RunnableLambda
    - RunnablePassthrough
    - StrOutputParser
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a PDF question-answering assistant.

Use ONLY the document context supplied below.

Rules:
1. Do not use outside knowledge.
2. If the answer is not supported by the context, reply exactly:
   I don't know
3. Keep the answer to a maximum of 150 words.
4. Be direct and factual.
5. Do not invent details, sources, page numbers, names, dates, or statistics.

Document context:
{context}
""",
            ),
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    retrieval_inputs = RunnableParallel(
        context=(
            RunnablePassthrough()
            | retriever
            | RunnableLambda(format_docs)
        ),
        question=RunnablePassthrough(),
    )

    chain = (
        retrieval_inputs
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def normalize_answer(answer: str) -> str:
    """
    Light safeguard for the assignment's 150-word maximum.
    The prompt should normally enforce this already.
    """
    answer = answer.strip()

    if answer.lower() in {
        "i don't know.",
        "i do not know.",
        "i don't know",
        "i do not know",
    }:
        return "I don't know"

    words = answer.split()
    if len(words) > 150:
        answer = " ".join(words[:150]).rstrip(".,;:") + "…"

    return answer


#
# Check for the API key before calling OpenAI services
#
# Stop early if the key is missing instead of failing later during embeddings or chat
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY was not found. Add it to a local `.env` file, "
        "then restart the Streamlit app."
    )
    st.code("OPENAI_API_KEY=your_openai_api_key_here")
    st.stop()


#
# Store processed document data across Streamlit reruns
#
# Session state avoids rebuilding the same vector store on every rerun
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "page_count" not in st.session_state:
    st.session_state.page_count = 0


#
# Sidebar controls for uploading a document
#
with st.sidebar:
    st.header("Document")

    # Only PDF files are accepted because the document reader is built for PDFs
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="The document is processed locally, while embeddings and answers use the OpenAI API.",
    )

    st.divider()

    st.markdown(
        """
**Pipeline**

PDF → Text → Chunks → Embeddings → FAISS → Top 3 chunks → ChatOpenAI
"""
    )


#
# Process the PDF only when a new file is uploaded
#
if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    current_hash = get_pdf_hash(pdf_bytes)

    # Hashing prevents the same uploaded PDF from being embedded more than once
    if current_hash != st.session_state.pdf_hash:
        with st.spinner("Extracting text, splitting chunks, and creating embeddings..."):
            try:
                page_documents = extract_pdf_pages(pdf_bytes)

                if not page_documents:
                    st.error(
                        "No extractable text was found in this PDF. "
                        "It may be image-only or scanned."
                    )
                    st.stop()

                chunks = split_documents(page_documents)
                vectorstore = build_vectorstore(chunks)

                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_hash = current_hash
                st.session_state.chunk_count = len(chunks)
                st.session_state.page_count = len(page_documents)

            except Exception as exc:
                st.exception(exc)
                st.stop()

    st.success(
        f"Ready: {st.session_state.page_count} text pages, "
        f"{st.session_state.chunk_count} chunks."
    )


#
# Retrieve relevant chunks and answer the user's question
#
if st.session_state.vectorstore is None:
    st.info("Upload a PDF to begin.")
    st.stop()

# Similarity search returns the three chunks most relevant to the question
retriever = st.session_state.vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Connect the retriever to the QA chain so answers stay grounded in the document
qa_chain = create_qa_chain(retriever)

st.subheader("Ask a question")

question = st.text_input(
    "Question",
    placeholder="Example: What are the main findings of the document?",
)

ask_clicked = st.button(
    "Ask",
    type="primary",
    use_container_width=True,
)

# Run retrieval and generation only after the user submits a question
if ask_clicked:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Searching the document and generating an answer..."):
            try:
                # Retrieve the supporting chunks separately so they can be shown
                # with the generated answer.
                source_documents = retriever.invoke(question)
                answer = qa_chain.invoke(question)
                answer = normalize_answer(answer)

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Source chunks")

                if not source_documents:
                    st.info("No source chunks were retrieved.")
                else:
                    for rank, doc in enumerate(source_documents, start=1):
                        page = doc.metadata.get("page", "Unknown")
                        chunk = doc.metadata.get("chunk", "Unknown")

                        with st.expander(
                            f"Source {rank} — Page {page}, Chunk {chunk}",
                            expanded=(rank == 1),
                        ):
                            st.write(doc.page_content)

            except Exception as exc:
                st.error("An error occurred while answering the question.")
                st.exception(exc)