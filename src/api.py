# api.py
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

# Nasz orchestrator RAG
from src.rag_pipeline import answer, ensure_all

app = FastAPI(title="RAG Demo", version="0.1.0")

# (opcjonalnie) CORS dla frontów uruchamianych z innych hostów/portów
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskBody(BaseModel):
    q: str

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/api/ask")
def api_ask_get(q: str = Query(..., min_length=1)):
    try:
        res = answer(q)
        return res
    except Exception as e:
        return {"answer": "", "hits": [], "message": str(e)}

@app.post("/api/ask")
def api_ask_post(body: AskBody):
    try:
        res = answer(body.q)
        return res
    except Exception as e:
        return {"answer": "", "hits": [], "message": str(e)}

@app.post("/api/reindex")
def api_reindex():
    # Wymuś (re)build indeksu i embeddingów; zwróć statystyki lub błąd
    try:
        stats = ensure_all()
        return {"stats": stats, "message": ""}
    except Exception as e:
        return {"stats": None, "message": str(e)}

# --- statyczne pliki (frontend) ---
STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)  # unikamy błędu "Directory 'static' does not exist"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
