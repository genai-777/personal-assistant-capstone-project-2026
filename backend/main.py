import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load .env from the project root (one level above backend/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Personal Productivity Assistant", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

orchestrator = Orchestrator()

# ── schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    task_type: str = ""

class PreferenceUpdate(BaseModel):
    value: str

# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html = Path(__file__).parent.parent / "frontend" / "index.html"
    if not html.exists():
        raise HTTPException(404, "Frontend not found")
    return HTMLResponse(html.read_text())


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")
    try:
        response  = orchestrator.chat(req.message)
        task_type = orchestrator._classify(req.message)   # for UI badge
        return ChatResponse(response=response, task_type=task_type)
    except Exception as e:
        logger.error(f"Orchestrator error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/reset")
async def reset():
    orchestrator.reset()
    return {"status": "conversation cleared"}


@app.get("/preferences")
async def get_prefs():
    return orchestrator.memory.get_preferences()


@app.patch("/preferences/{key}")
async def update_pref(key: str, body: PreferenceUpdate):
    orchestrator.memory.update_preference(key, body.value)
    return {"status": "updated", "key": key, "value": body.value}


@app.get("/history")
async def history(limit: int = 10):
    return orchestrator.memory.get_recent_tasks(limit)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agents": ["Orchestrator","EmailAgent","ResearchAgent",
                   "MeetingBriefAgent","SafetyAgent"],
        "tools":  ["Gmail","GoogleCalendar","TavilySearch"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",
                host=os.getenv("APP_HOST", "0.0.0.0"),
                port=int(os.getenv("APP_PORT", 8000)),
                reload=True)
