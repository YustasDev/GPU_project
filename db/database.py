from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# Строка подключения (Connection String)
# Формат: диалект+драйвер://имя:пароль@хост:порт/имя_базы (вам надо поставить свои данные)
DATABASE_URL = "postgresql+psycopg2://myuser:mypassword@localhost:5432/mydatabase"


# Создаем "Двигатель" (Engine) - точку входа в БД
# echo=False скрывает сырые SQL запросы из консоли. Для дебага можно поставить True.
engine = create_engine(DATABASE_URL, echo=False)

# SessionLocal - это фабрика сессий. Сессия - это "разговор" с базой данных (транзакция)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)

# Базовый класс, от которого мы будем наследовать все наши таблицы
class Base(DeclarativeBase):
    pass