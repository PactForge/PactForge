from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from google import genai
from google.genai import types
import time
from docx import Document
import os

# Load your API key from environment or config
import os
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)
model_config = types.GenerateContentConfig(temperature=0.75, top_p=0.9)

# Initialize FastAPI app
app = FastAPI()

# Allow frontend access (adjust origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load documents
DATA_PATH = "./Clauses"
collections = {}
clientdb = chromadb.Client()
agreement_types = ["rent", "nda", "employment", "franchise", "contractor"]

for name in agreement_types:
    col = clientdb.get_or_create_collection(name=f"{name}_agreements")
    clauses = []
    path = f"{DATA_PATH}/{name}.docx"
    doc = Document(path)
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            clauses.append(text)
    embeds, ids, docs = [], [], []
    for i, clause in enumerate(clauses):
        embed = client.models.embed_content(
            model="models/text-embedding-004",
            contents=clause,
            config=types.EmbedContentConfig(task_type="retrieval_document")
        )
        time.sleep(0.4)
        embeds.append(embed.embeddings[0].values)
        ids.append(f"{name}-{i}")
        docs.append(clause)
    col.add(embeddings=embeds, ids=ids, documents=docs)
    collections[name] = col

class AgreementInput(BaseModel):
    agreement_type: str
    important_info: str
    extra_info: str

@app.post("/generate")
async def generate_agreement(data: AgreementInput):
    user_input = data.important_info + "\n" + data.extra_info
    embed = client.models.embed_content(
        model="models/text-embedding-004",
        contents=user_input,
        config=types.EmbedContentConfig(task_type="retrieval_query")
    )
    query_embedding = embed.embeddings[0].values
    db = collections[data.agreement_type]
    results = db.query(query_embeddings=query_embedding, n_results=5)
    relevant_docs = results['documents']

    # Load sample agreements
    with open("./sampleagreements/sample.txt", "r") as f:
        sample_agreements = f.read()

    prompt = f"""
        You are a helpful AI assistant for law agreement generation.

        The agreement type is: {data.agreement_type}
        Important information: {data.important_info}
        Additional input: {data.extra_info}

        Relevant clauses:
        {relevant_docs}

        Format the agreement similar to:
        {sample_agreements}

        Make it clear, complete, and legally sound. Output only the agreement text.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=model_config
    )
    return {"agreement": response.text}
