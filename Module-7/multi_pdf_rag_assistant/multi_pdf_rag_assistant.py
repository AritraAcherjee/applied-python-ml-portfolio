"""Engineering-style multi-PDF RAG assistant for the AI evolution assignment."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatopenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_QUERY = "Explain the evolution of AI in a few sentences."


@dataclass(frozen=True)
class Settings:
    """All project settings in one validated object."""

    pdf_paths: tuple[Path, ...]
    chunk_size: int = 300
    chunk_overlap: int = 50
    retrieval_k: int = 4
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    index_directory: Path = Path("faiss_index")

    def validate(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and add your key."
            )
        missing = [str(path) for path in self.pdf_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing required PDF file(s): " + ", ".join(missing))
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")


class MultiPDFRAGAssistant:
    """Loads PDFs, builds a FAISS index, and answers grounded questions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        self.vector_store: FAISS | None = None
        self.qa_chain: RetrievalQA | None = None

    def load_documents(self) -> list[Any]:
        documents: list[Any] = []
        for pdf_path in self.settings.pdf_paths:
            pages = PyPDFLoader(str(pdf_path)).load()
            if not pages:
                raise ValueError(f"No readable text was found in {pdf_path.name}.")
            for page in pages:
                page.metadata["source"] = pdf_path.name
            documents.extend(pages)
            logging.info("Loaded %d page(s) from %s", len(pages), pdf_path.name)
        return documents

    def split_documents(self, documents: list[Any]) -> list[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise ValueError("Document splitting produced no chunks.")
        for number, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = number
        logging.info("Created %d chunks (size=%d, overlap=%d)", len(chunks), self.settings.chunk_size, self.settings.chunk_overlap)
        return chunks

    def build_vector_store(self, chunks: list[Any]) -> FAISS:
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.settings.index_directory.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.settings.index_directory))
        logging.info("Saved FAISS index to %s", self.settings.index_directory)
        return self.vector_store

    def build_qa_chain(self) -> RetrievalQA:
        if self.vector_store is None:
            raise RuntimeError("Build the FAISS vector store before the QA chain.")

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.settings.retrieval_k},
        )
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a document-grounded AI assistant. Use only the supplied "
                "context. Combine the historical development and future direction of AI "
                "when both are relevant. Answer in a few concise, coherent sentences. "
                "If the context is insufficient, say so.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            ),
        )
        llm = ChatOpenAI(model=self.settings.chat_model, temperature=self.settings.temperature)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        return self.qa_chain

    def ask(self, query: str) -> dict[str, Any]:
        if self.qa_chain is None:
            raise RuntimeError("Build the QA chain before submitting a query.")
        result = self.qa_chain.invoke({"query": query})
        answer = result.get("result", "").strip()
        sources = result.get("source_documents", [])
        if not answer:
            raise RuntimeError("The model returned an empty answer.")
        return {"answer": answer, "sources": sources]

    def run(self, query: str) -> dict[str, Any]:
        self.settings.validate()
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        self.build_vector_store(chunks)
        self.build_qa_chain()
        return self.ask(query)


def display_result(query: str, result: dict[str, Any]) -> None:
    print("\nQUESTION")
    print(query)
    print("\nANSWER")
    print(result["answer"])
    print("\nRETRIEVED SOURCES")

    source_names: set[str] = set()
    for position, document in enumerate(result["sources"], start=1):
        source = str(document.metadata.get("source", "Unknown source")
        page = int(document.metadata.get("page", 0)) + 1
        source_names.add(source)
        preview = "".join(document.page_content.split())[:180]
        print(f"{position}. {source}, page {page}: {preview}...")
    expected = {path.name for path in SETTINGS.pdf_paths}
    if expected.issubset(source_names):
        print("\nVALIDATION: The retrieved context includes both PDFs.")
    else:
        missing = ", ".join(sorted(expected - source_names))
        print(f"\nVALIDATION WARNING: Top-4 retrieval did not include: {missing}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question across two AI PDFs.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Question sent to the RAG pipeline.")
    return parser.parse_args()


SETTINGS = Settings(
    pdf_paths=(Path("history_of_ai.pdf"), Path("future_of_ai.pdf")),
)


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = parse_arguments()
        result = MultiPDFRAGAssistant(SETTINGS).run(args.query)
        display_result(args.query, result)
        return 0
    except (EnvironmentError, FileNotFoundError, ValueError, RuntimeError) as error:
        logging.error("%s", error)
        return 1
    except Exception as error:
        logging.exception("Unexpected failure: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main()
