#  RAG-based Medical Q&A Chatbot (Streamlit + FastAPI + Groq + LangChain)

A medical question-answering system built using **RAG (Retrieval-Augmented Generation)**.

---

##  Features
- Upload PDF / DOCX / TXT medical documents
- Automatic text extraction
- Semantic search using FAISS
- Llama 3.1 (Groq) for answer generation
- Streamlit UI + FastAPI backend

---

## Project Structure
.RAG-based Medical Q&A Chatbot
├── app.py            # Streamlit UI
├── main.py           # FastAPI backend
├── model.py          # RAG + LLM logic
├── requirements.txt
└── README.md


##  Installation

```bash
git clone <repo-url>
cd RAG-based Medical Q&A Chatbot
pip install -r requirements.txt


##Environment Variables

Create .env:
GROQ_API_KEY=your_key

## Run FastAPI Backend

uvicorn main:app --reload

## Run Streamlit App

streamlit run app.py
Make sure FastAPI server is running before using Streamlit.

## How It Works

Upload a medical document
Text is extracted → chunked → embedded
Query is semantically searched
Groq LLM generates answer using context

##Model Used

Embedding: all-MiniLM-L6-v2
LLM: llama-3.1-8b-instant

## Author
Created by Muskan (GenAI Engineer)
