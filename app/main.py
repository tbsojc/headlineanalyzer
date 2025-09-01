# ============================
# 📁 app/main.py
# (aus deinem bisherigen main.py)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import router as api_router
from fastapi_utils.tasks import repeat_every
from app.core.feeds import fetch_articles
from app.models import save_articles
from app.api.visuals import router as visuals_router




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
@repeat_every(seconds=1800)  # alle 30 Minuten
def scheduled_refresh() -> None:
    print("[INFO] ⏱ Auto-refresh gestartet...")
    articles = fetch_articles()
    save_articles(articles)
    print(f"[INFO] ✅ {len(articles)} Artikel aktualisiert.")


app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")

app.include_router(api_router)
app.include_router(visuals_router)
