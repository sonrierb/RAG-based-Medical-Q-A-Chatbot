import os
import uuid
from typing import List
from groq import Groq
from dotenv import load_dotenv
import os



from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class MedicalRAGSystem:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.vector_store = None
        self.current_doc_id = None
        
    def extract_text_from_file(self, file_path: str, file_type: str) -> str:
        try:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_type == "docx":
                loader = Docx2txtLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
                
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
        except Exception as e:
            return f"Error extracting text: {str(e)}"
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        return documents
    
    def create_vector_store(self, documents, doc_id: str):
        self.vector_store = FAISS.from_documents(documents, self.embedding_model)
        self.current_doc_id = doc_id
        return True
    
    def semantic_search(self, query: str, k: int = 4):
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)
    
    def generate_answer(self, query: str, context: str):
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a medical assistant. Based on the provided medical context, answer the user's question accurately.

Medical Context:
{context}

User Question: {question}

Instructions:
1. Answer based ONLY on the given context.
2. If answer is not present, say: "I don't have enough information."
3. Always advise consulting a healthcare professional.

Answer:
"""
        )
        
        try:
            formatted_prompt = prompt_template.format(
                context=context, question=query
            )

            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": formatted_prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=1024
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def process_query(self, query: str):
        relevant_docs = self.semantic_search(query)
        
        if not relevant_docs:
            return "I don't have enough information. Please upload and process a document."
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        return self.generate_answer(query, context)

