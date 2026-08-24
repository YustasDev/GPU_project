import os  # для чтения переменных окружения

from dotenv import load_dotenv  # подгрузка .env при локальном запуске
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

load_dotenv()  # подхватываем .env, если он есть (актуально при локальном запуске вне Docker)

# Строка подключения (Connection String)
# Формат: диалект+драйвер://имя:пароль@хост:порт/имя_базы
# Секрет НЕ хардкодим НИГДЕ (даже как «запасной вариант»): строку целиком берём
# из переменной окружения DATABASE_URL — локально из .env, в Docker её подставляет
# docker-compose (там хост — "db").
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:  # переменной нет — честно падаем с понятной ошибкой, а не с дефолтным паролем в коде
    raise RuntimeError(
        "Переменная окружения DATABASE_URL не задана. "
        "Скопируйте .env.example в .env и заполните его (см. главу 12)."
    )


# Создаем "Двигатель" (Engine) - точку входа в БД
# echo=False скрывает сырые SQL запросы из консоли. Для дебага можно поставить True.
engine = create_engine(DATABASE_URL, echo=False)

# SessionLocal - это фабрика сессий. Сессия - это "разговор" с базой данных (транзакция)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,  # после commit() объект не обесценивается: record.id остаётся
                             # в памяти и не вызывает повторный SELECT той же строки
    bind=engine,
)

# Базовый класс, от которого мы будем наследовать все наши таблицы
class Base(DeclarativeBase):
    pass