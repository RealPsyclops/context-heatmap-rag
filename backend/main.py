from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional
import time
import uuid
import numpy as np

# Import our custom modules
from .models import Message, Anchor, HeatSignal
from .retriever import ThermalRetriever

app = FastAPI(title="Context Heatmap RAG")

# --- IN-MEMORY DATABASE (Replace with Postgres/VectorDB in Prod) ---
DB_MESSAGES: List[Message] = []
DB_ANCHORS: List[Anchor] = []

# --- MOCK EMBEDDING FUNCTION ---
# In prod, replace this with OpenAI/Cohere embeddings
def get_embedding(text: str) -> List[float]:
    # Returns a random normalized vector for demo purposes
    vec = np.random.rand(1536)
    return (vec / np.linalg.norm(vec)).tolist()

# --- INITIALIZATION ---
@app.on_event("startup")
def seed_data():
    """Seeds the chat with a dummy conversation for testing."""
    print("🌱 Seeding database with dummy conversation...")
    conversation = [
        ("user", "I need to fix the auth middleware."),
        ("assistant", "Here is the Python code for JWT auth: `def verify_token()...`"),
        ("user", "It's throwing a 403 error."),
        ("assistant", "Check your scope definitions. Here is the fixed scope list."),
    ]
    for role, content in conversation:
        msg = Message(
            id=str(uuid.uuid4()),
            content=content,
            role=role,
            embedding=get_embedding(content)
        )
        DB_MESSAGES.append(msg)

# --- Pydantic Models for API Requests ---
class ChatRequest(BaseModel):
    query: str

class SignalRequest(BaseModel):
    message_id: str
    text_snippet: str
    signal_type: str  # 'highlight', 'copy', 'hover'
    start_index: int = 0
    end_index: int = 0

class AnchorRequest(BaseModel):
    message_id: str
    user_label: str

# --- ENDPOINTS ---

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Standard Chat Endpoint.
    1. Embeds query.
    2. Runs Thermal Retrieval (Heatmap + Anchors).
    3. Returns relevant context.
    """
    query_vec = get_embedding(req.query)
    
    # Initialize Retriever with current state
    retriever = ThermalRetriever(DB_MESSAGES, DB_ANCHORS)
    
    # Retrieve top 3 chunks
    results = retriever.retrieve(query_vec, top_k=3)
    
    # In a real app, you would feed 'results' to GPT-4 here.
    # We just return the context to visualize what the LLM *would* see.
    return {
        "reply": "I found some context based on your heat signals.",
        "retrieved_context": results
    }

@app.post("/signal")
async def receive_signal(sig: SignalRequest):
    """
    The Passive Listener Endpoint.
    Receives 'selectionchange' or 'copy' events from frontend.
    """
    # Find the message
    target_msg = next((m for m in DB_MESSAGES if m.id == sig.message_id), None)
    if not target_msg:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Logic: Add the range to the message's heat map
    # In a real DB, you'd store this in a separate 'signals' table.
    target_msg.heat_ranges.append((sig.start_index, sig.end_index))
    
    weight = 2.0 if sig.signal_type == 'copy' else 1.0
    print(f"🔥 HEATING UP Message {sig.message_id[:8]}... (Type: {sig.signal_type}, Weight: {weight})")
    
    return {"status": "success", "heat_score_updated": True}

@app.post("/anchor")
async def create_anchor(req: AnchorRequest):
    """
    The 'Save to Library' Endpoint.
    Creates a side-door entry for specific retrieval.
    """
    target_msg = next((m for m in DB_MESSAGES if m.id == req.message_id), None)
    if not target_msg:
        raise HTTPException(status_code=404, detail="Message not found")
    
    new_anchor = Anchor(
        id=str(uuid.uuid4()),
        source_message_id=req.message_id,
        user_label=req.user_label,
        label_embedding=get_embedding(req.user_label) # Embed the LABEL, not the content
    )
    
    DB_ANCHORS.append(new_anchor)
    print(f"⚓ ANCHOR DROPPED: '{req.user_label}' -> Message {req.message_id[:8]}")
    
    return {"status": "success", "anchor_id": new_anchor.id}

@app.get("/debug")
async def get_state():
    """Helper to visualize the current state of the DB."""
    return {
        "total_messages": len(DB_MESSAGES),
        "total_anchors": len(DB_ANCHORS),
        "messages": DB_MESSAGES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
