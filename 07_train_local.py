from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    # Обязательная защита для Windows/WSL2 при многопроцессорности
    multiprocessing.freeze_support()

    # 1. Если для обучения вы используете GPU с 8 ГБ VRAM лучше взять версию Small,
    # чтобы уместить в память размер батча побольше.
    model = YOLO("yolo11s.pt")

    # 2. Запускаем процесс обучения
    print("[INFO] Старт обучения...")
    results = model.train(
        data="/home/yustasdev/datasets/cow_dataset/data.yaml",  # Путь к нашему конфигу
        epochs=50,  # Количество эпох (проходов по датасету)
        imgsz=640,  # Размер картинки (ресайз перед подачей)
        batch=8,    # Размер батча (сколько картинок грузить в VRAM за раз)
        device="cuda:0",  # Обучать на GPU
        workers=4,  # Сколько потоков CPU готовят данные
        project="ai_runs",  # Папка для сохранения результатов
        name="cow_learning"  # Название эксперимента
    )

    print("[INFO] Обучение завершено!")