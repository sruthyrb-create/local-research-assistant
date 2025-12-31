`import os
import json
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS 
from langchain_community.chat_models import ChatOllama 
from langchain_core.documents import Document 
from langchain_core.runnables import RunnablePassthrough,RunnableLambda 
from langchain_ollama import ChatOllama

class ResearchAssistant:

    def __init__(self, docs_path="docs", index_path="faiss_index", rebuild_index=False):
        self.docs_path = docs_path
        self.index_path = index_path

        # Embeddings model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load or rebuild FAISS
        if rebuild_index or not os.path.exists(index_path):
            print("Building FAISS index from documents...")
            self._build_faiss_index()
        else:
            print("Loading existing FAISS index...")
            self.vectorstore = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Loaded FAISS index with ~{len(self.vectorstore.index_to_docstore_id)} chunks.")

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # Initialize RAG chain
        self._init_rag_chain()

    # ---------------------------------------------------------
    # Build FAISS index from local documents
    # ---------------------------------------------------------
    def _build_faiss_index(self):
        documents = []

        for root, _, files in os.walk(self.docs_path):
            for file in files:
                if file.endswith(".txt") or file.endswith(".md"):
                    path = os.path.join(root, file)
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()

                    documents.append({
                        "source": file,
                        "content": content
                    })

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        all_chunks = []
        for doc in documents:
            chunks = splitter.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "text": chunk,
                        "metadata": {"source": doc["source"], "chunk": i}
                    }
                )

        # Convert to FAISS format
        texts = [c["text"] for c in all_chunks]
        metadatas = [c["metadata"] for c in all_chunks]

        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )

        # Save index
        self.vectorstore.save_local(self.index_path)
        print(f"FAISS index built with {len(texts)} chunks.")

    # ---------------------------------------------------------
    # Build RAG LCEL chain
    # ---------------------------------------------------------
    def _init_rag_chain(self):
        llm = ChatOllama(model="llama3")

        # Format retrieved context into a prompt
        def format_prompt(inputs):
            context_text = ""
            for doc in inputs["context"]:
                context_text += (
                    f"\n---\n"
                    f"Source: {doc.metadata['source']} (chunk {doc.metadata['chunk']})\n"
                    f"{doc.page_content}\n"
                )

            return (
                "You are a highly intelligent research assistant.\n"
                "Use ONLY the retrieved context to answer the question.\n"
                f"{context_text}\n"
                f"Question: {inputs['question']}\n"
                "Answer clearly and concisely.\n"
            )

        # LCEL pipeline
        self.qa_chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough()
            }
            | RunnableLambda(format_prompt)
            | llm
        )

    # ---------------------------------------------------------
    # Ask a question using RAG
    # ---------------------------------------------------------
        # -------------------------------------------------------------
   
    def ask(self, question: str, top_k: int = 4):
        # update retriever's top-k value
        self.retriever.search_kwargs["k"] = top_k

        response = self.qa_chain.invoke(question)
        answer = response.content if hasattr(response, "content") else str(response)

        # Collect retrieved docs for transparency
        docs = self.retriever._get_relevant_documents(question, run_manager=None)
        sources = [
            {"source": d.metadata["source"], "chunk": d.metadata["chunk"]}
            for d in docs
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }



# For direct testing
if __name__ == "__main__":
    ra = ResearchAssistant(rebuild_index=False)
    print(ra.ask("What is inside the documents?"))
`