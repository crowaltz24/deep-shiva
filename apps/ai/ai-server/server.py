from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple

from apps.ai.rag.chat import _load_system_prompt, _format_history
from apps.ai.rag.ingest import get_rag_index
from apps.ai.rag.llm_setup import configure_llamaindex
from apps.ai.rag.utils import get_weather_data_for_place
from llama_index.core import PromptTemplate

query_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global query_engine
    print("AI Server is starting up...")
    
    print("Configuring LlamaIndex...")
    configure_llamaindex()
    
    print("Loading RAG index from storage...")
    # build/load index
    index = get_rag_index()
    print("RAG index loaded.")
    
    print("Creating RAG query engine...")
    system_prompt = _load_system_prompt()
    text_qa_template = PromptTemplate(
        (
            f"{system_prompt}\n\n"
            "Given the following context, answer the user's question.\n"
            "- If context is not relevant, say so briefly or answer succinctly from general knowledge.\n\n"
            "Context:\n{context_str}\n\n"
            "Question: {query_str}\n\n"
            "Answer:"
        )
    )
    query_engine = index.as_query_engine(
        streaming=False,
        similarity_top_k=5,
        text_qa_template=text_qa_template,
    )
    
    print("Startup complete. AI Engine is ready.")
    # Ending 
    yield
    
    print("AI Server is shutting down.")

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    query: str
    
@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not query_engine:
        return {"error": "Query engine is not initialized"}, 503

    try:
        response = query_engine.query(request.query)
        answer = getattr(response, 'response', str(response))
        return {"response": answer}
    except Exception as e:
        return {"error": str(e)}, 500

@app.get("/")
def root():
    return {"status": "huhhhaaha"}

def read_root():
    return {"status": "AI server is running"}