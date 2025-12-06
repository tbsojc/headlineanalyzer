from pydantic import BaseModel
from datetime import datetime

class Article(BaseModel):
    id: int | None = None
    title: str
    teaser: str | None = None
    url: str
    source: str
    topic: str
    tags: list[str] | None = None
    published_at: datetime

    class Config:
        from_attributes = True  # SQLAlchemy -> Pydantic
