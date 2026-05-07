from sqlalchemy import String, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from database import Base

class Detection(Base):
    __tablename__ = "detections"  # Имя таблицы в PostgreSQL

    # Идентификатор (Primary Key), создается автоматически (1, 2, 3...)
    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Время события. По умолчанию ставим текущее время сервера
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Что мы нашли (например, "person", "car")
    object_class: Mapped[str] = mapped_column(String(50), index=True)

    # Насколько уверена нейросеть в детектировании объекта (от 0 до 1)
    confidence: Mapped[float] = mapped_column(Float)

    # Координаты рамки в формате JSON-строки "[x1, y1, x2, y2]"
    bounding_box: Mapped[str] = mapped_column(String(100))

    # Путь к сохраненному скриншоту на диске
    image_path: Mapped[str] = mapped_column(String(255), nullable=True)

    def __repr__(self):
        # Это для красивого вывода объекта в консоль (print)
        return f"<Detection(id={self.id}, class='{self.object_class}', conf={self.confidence:.2f})>"