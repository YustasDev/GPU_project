# api/main_api.py
from fastapi import FastAPI, Request, Depends  # ядро фреймворка + Request (нужен Jinja2) + Depends (DI для get_db)
from fastapi.staticfiles import StaticFiles  # отдача статических JPEG из data/saved_events
from fastapi.templating import Jinja2Templates  # рендер index.html с подстановкой событий из БД
from sqlalchemy.orm import Session  # тип сессии БД — нужен только для аннотации параметров эндпоинтов
from pydantic import BaseModel, ConfigDict  # схема ответа EventOut для /api/events (Swagger покажет структуру JSON)
from datetime import datetime  # тип поля timestamp в схеме EventOut
import os
import sys
from pathlib import Path

# Корень проекта вычисляем от расположения этого файла — независимо от текущей рабочей директории
# чтобы uvicorn можно было запускать из любой директории
PROJECT_DIR = Path(__file__).resolve().parent.parent  # ../.. от api/main_api.py
SAVED_EVENTS_DIR = PROJECT_DIR / "data" / "saved_events"  # тот же каталог, куда пишет core/logger.py
TEMPLATES_DIR = PROJECT_DIR / "api" / "templates"  # каталог Jinja2-шаблонов рядом с этим модулем

# sys.path.append нужен, чтобы импорты `from db...` работали при запуске `uvicorn api.main_api:app` —
# uvicorn кладёт в sys.path только пакет api, а корень проекта (где лежит db/) — нет.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import SessionLocal  # фабрика сессий БД (см. db/database.py)
from db.models import Detection  # ORM-модель таблицы detections; строки в неё пишет core/logger.py


# Pydantic-схема ответа для /api/events. from_attributes=True разрешает Pydantic читать поля
# прямо из ORM-объекта Detection (в Pydantic v1 это называлось orm_mode). Благодаря этой схеме
# в response_model Swagger (/docs) показывает структуру JSON, а не безымянный пустой ответ.
class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # читать атрибуты ORM-объекта, а не только dict

    id: int  # первичный ключ строки detections
    timestamp: datetime  # время события (UTC); FastAPI сериализует в ISO-строку
    object_class: str  # имя класса: "person" / "car"
    confidence: float  # уверенность нейросети, 0..1
    bounding_box: str  # координаты рамки строкой "[x1, y1, x2, y2]"
    image_path: str | None = None  # путь к JPEG; в модели nullable, поэтому Optional


app = FastAPI(title="Smart Observer API")  # экземпляр приложения; uvicorn находит его по имени `app`

# Раздача JPEG-скриншотов. Каждое событие из core/logger.py создаёт файл в SAVED_EVENTS_DIR;
# этот mount превращает их в URL вида /images/<filename> для тега <img> в шаблоне.
SAVED_EVENTS_DIR.mkdir(parents=True, exist_ok=True)  # подстраховка: каталог обычно уже создан DBLogger
app.mount("/images", StaticFiles(directory=str(SAVED_EVENTS_DIR)), name="images")  # name="images" — для url_for, если понадобится

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))  # одна точка рендера шаблонов на всё приложение


# FastAPI Dependency: открываем сессию DB на каждый запрос и гарантированно закрываем.
# Через generator + try/finally close() сработает даже при исключении в хендлере.
def get_db():
    db = SessionLocal()  # отдельная сессия на каждый HTTP-запрос — без shared state между запросами
    try:
        yield db  # отдаём сессию хендлеру; всё после yield выполнится уже после возврата ответа
    finally:
        db.close()  # finally — close даже при исключении внутри хендлера (иначе утечка коннектов)


# JSON-эндпоинт для машинных клиентов (curl, скрипты, fetch с фронта).
# По умолчанию отдаёт 10 свежих событий; limit меняется через query string: /api/events?limit=50.
@app.get("/api/events", response_model=list[EventOut])
async def get_events_json(limit: int = 10, db: Session = Depends(get_db)):
    # SELECT * FROM detections ORDER BY timestamp DESC LIMIT :limit — новейшие сверху
    events = db.query(Detection).order_by(Detection.timestamp.desc()).limit(limit).all()
    return events  # FastAPI прогонит каждый ORM-объект через EventOut (from_attributes) и отдаст JSON


# Главная страница дашборда — HTML с Bootstrap-сеткой карточек.
# Лимит 20 захардкожен; фильтры/пагинация добавятся позже, если понадобятся.
@app.get("/")
async def serve_dashboard(request: Request, db: Session = Depends(get_db)):
    events = db.query(Detection).order_by(Detection.timestamp.desc()).limit(20).all()  # тот же запрос, что и в /api/events, но limit=20
    # Современная сигнатура Starlette: request передаём ПЕРВЫМ позиционным аргументом, а в контексте
    # оставляем только свои данные. Старая форма TemplateResponse(name, {"request": ...}) — deprecated.
    return templates.TemplateResponse(request, "index.html", {"events": events})
