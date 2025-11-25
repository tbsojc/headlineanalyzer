# ============================
# 📁 app/main.py

from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse, HTMLResponse
from starlette.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path


from fastapi_utils.tasks import repeat_every

# API-Router
from app.api.routes import router as api_router
from app.api.visuals import router as visuals_router

# Feeds (liefert ArticleSchema-Objekte)
from app.core.feeds import fetch_articles

# ORM / DB / Repository
from app.database import Base, engine, SessionLocal
from app.models_sql import ArticleORM  # sicherstellen, dass das Model registriert ist
from app.repositories.articles import bulk_upsert_articles


app = FastAPI(default_response_class=ORJSONResponse)

# --- GZip (Antworten ab 1 KB komprimieren) ---
app.add_middleware(GZipMiddleware, minimum_size=1024)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static files für Assets (CSS, Bilder etc.) ---
app.mount("/static", StaticFiles(directory="app/static", html=False), name="static")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/ui/", response_class=HTMLResponse)
async def ui_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ui/{keyword}", response_class=HTMLResponse)
async def ui_detail(request: Request, keyword: str):
    return templates.TemplateResponse("keyword.html", {
        "request": request,
        "keyword": keyword,
    })


# --- Router ---
app.include_router(api_router)
app.include_router(visuals_router)


# --- Tabellen anlegen (einmalig beim Start, falls keine Migrationen verwendet werden) ---
@app.on_event("startup")
def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    print("[INFO] 📦 Datenbanktabellen geprüft/erstellt.")


# --- Geplanter Refresh-Job (alle 30 Minuten) ---
@app.on_event("startup")
@repeat_every(seconds=1800)  # 30 Minuten
def scheduled_refresh() -> None:
    print("[INFO] ⏱ Auto-Refresh gestartet...")
    items = fetch_articles()  # -> List[ArticleSchema]
    with SessionLocal() as db:
        try:
            count = bulk_upsert_articles(db, items)
            print(f"[INFO] ✅ {count} neue Artikel gespeichert (Duplikate wurden ignoriert).")
        except Exception as e:
            print(f"[ERROR] Refresh fehlgeschlagen: {e}")
            raise
