from fastapi import FastAPI, UploadFile
import uuid
import tempfile
from model import MedicalRAGSystem

app = FastAPI()
rag = MedicalRAGSystem()


@app.post("/process-document")
async def process_document(file: UploadFile):
    doc_id = str(uuid.uuid4())[:8]
    ext = file.filename.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(await file.read())
        path = tmp.name

    text = rag.extract_text_from_file(path, ext)

    if text:
        docs = rag.chunk_text(text)
        rag.create_vector_store(docs, doc_id)

    return {"status": "success", "doc_id": doc_id}


@app.post("/ask")
async def ask_question(question: str):
    answer = rag.process_query(question)
    return {"answer": answer}
