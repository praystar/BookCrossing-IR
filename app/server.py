"""FastAPI backend + static frontend."""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import FRONTEND_DIR, METHODS, RESULTS_DIR
from .engine import SearchEngine

state: dict = {"engine": None, "error": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state["engine"] = SearchEngine.load()
        print("Index loaded:", state["engine"].meta.get("encoder"))
    except Exception as exc:  # keep the server up and explain what to do
        state["error"] = str(exc)
        print("[error]", exc)
    yield


app = FastAPI(title="Book Search (Book-Crossing IR)", version="1.0", lifespan=lifespan)


def _engine() -> SearchEngine:
    if state["engine"] is None:
        raise HTTPException(503, state["error"] or "Index not loaded. Run `python run.py build`.")
    return state["engine"]


@app.get("/api/health")
def health():
    return {"status": "ok" if state["engine"] else "no-index", "error": state["error"]}


@app.get("/api/info")
def info():
    return _engine().info()


@app.get("/api/search")
def search(q: str = Query("", description="Free-text query"),
           method: str = Query("hybrid", description="tfidf | bm25 | semantic | hybrid"),
           k: int = Query(10, ge=1, le=50)):
    if method not in METHODS:
        raise HTTPException(400, f"method must be one of {list(METHODS)}")
    return _engine().search(q, method, k)


@app.get("/api/compare")
def compare(q: str = Query(""), k: int = Query(10, ge=1, le=25)):
    eng = _engine()
    return {"query": q, "runs": {m: eng.search(q, m, k) for m in METHODS}}


@app.get("/api/metrics")
def metrics():
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "No evaluation yet. Run `python run.py evaluate`.")
    return json.loads(path.read_text())


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
