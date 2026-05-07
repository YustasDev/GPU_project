from database import engine, SessionLocal, Base
from models import Detection
from datetime import datetime

# 1. МАГИЯ: Эта команда смотрит на модели (наследованные от Base)
# и отправляет в Postgres команду CREATE TABLE... если таблицы еще нет.
print("Создаем таблицы в базе данных...")
Base.metadata.create_all(bind=engine)
print("Таблицы готовы!\n")


def create_mock_detection():
    # Открываем сессию (транзакцию)
    with SessionLocal() as db:
        # 2. Создаем Python-объект нашего события
        new_event = Detection(
            object_class="person",
            confidence=0.98,
            bounding_box="[100, 150, 400, 500]",
            image_path="/data/images/event_001.jpg"
            # timestamp сгенерируется автоматически
        )

        # 3. Добавляем в сессию и фиксируем (сохраняем) в БД
        db.add(new_event)
        db.commit()

        # Обновляем объект, чтобы получить его ID из базы
        db.refresh(new_event)
        print(f"Успешно сохранено! ID записи: {new_event.id}")


def read_detections():
    with SessionLocal() as db:
        # 4. Читаем из БД: Дай мне все записи, где уверенность > 0.90
        # И отсортируй по времени убывания (новые сверху)
        print("\n Чтение из базы данных:")

        results = db.query(Detection).filter(
            Detection.confidence > 0.90
        ).order_by(Detection.timestamp.desc()).limit(5).all()

        for item in results:
            print(f"Событие: {item.object_class} (Уверенность: {item.confidence * 100:.1f}%) в {item.timestamp}")


if __name__ == "__main__":
    # Симулируем 3 детекции подряд
    for _ in range(3):
        create_mock_detection()

    # Читаем результат
    read_detections()