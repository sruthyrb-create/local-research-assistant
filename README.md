# 🧠 Local Research Assistant (RAG System)

A local Retrieval-Augmented Generation (RAG) research assistant that answers questions using your own documents with full source transparency.

## 🔹 Features
- Semantic document retrieval using FAISS
- Sentence-transformer embeddings
- LLaMA-powered answer generation via Ollama
- Source-aware responses with document references
- Interactive CLI mode

## 🔹 Tech Stack
- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- Ollama (LLaMA)
- Retrival-Augmented Generation(RAG)

## 🔹 Project Structure

data/ # Input documents
faiss_index/ # Vector index
research_assistant.py
main.py
demo_outputs.json


## 🔹 How to Run

```bash
pip install -r requirements.txt
ollama run llama3
python main.py
