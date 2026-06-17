import os
import vertexai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import firestore
from vertexai.language_models import TextEmbeddingModel
from google.cloud.aiplatform import MatchingEngineIndexEndpoint
import vertexai.generative_models as genai

# ── Config
PROJECT_ID = "project-d533213a-0f7d-4f94-94e"
REGION = "us-central1"
ENDPOINT_ID = "projects/414964344961/locations/us-central1/indexEndpoints/640876740528308224"
DEPLOYED_INDEX_ID = "index_deployed_1781630780245"

# ── Init
vertexai.init(project=PROJECT_ID, location=REGION)
db = firestore.Client(project=PROJECT_ID, database="database")
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
endpoint = MatchingEngineIndexEndpoint(index_endpoint_name=ENDPOINT_ID)

gemini = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: QueryRequest):
    # 1. Embed the question
    q_embedding = embed_model.get_embeddings([req.question])[0].values

    # 2. Query Vector Search
    results = endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[q_embedding],
        num_neighbors=req.top_k,
    )

    # 3. Fetch chunk text from Firestore
    chunk_ids = [n.id for n in results[0]]
    if not chunk_ids:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    chunks = []
    for chunk_id in chunk_ids:
        doc = db.collection("chunk_metadata").document(chunk_id).get()
        if doc.exists:
            chunks.append(doc.to_dict().get("text", ""))

    context = "\n\n".join(chunks)

    # 4. Call Gemini
    prompt = f"""You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {req.question}
"""
    response = gemini.generate_content(prompt)

    return {
        "question": req.question,
        "answer": response.text,
        "chunks_used": len(chunks),
    }