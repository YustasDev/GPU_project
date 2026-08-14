from threading import Thread
import cv2
import time
from queue import Queue, Empty

class FileVideoStream:
    def __init__(self, path, queue_size=128):
        # Инициализация
        self.stream = cv2.VideoCapture(path)

        # Сразу проверяем, что источник открылся. Без этой проверки опечатка в имени
        # файла или занятая другой программой камера выглядят как "видео закончилось"
        if not self.stream.isOpened():
            raise IOError(f"Не удалось открыть источник видео: {path!r}")

        # Метаданные читаем ОДИН раз и запоминаем. Спрашивать их у self.stream позже
        # опасно: к тому моменту чтение может быть уже завершено
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        # ХАК ДЛЯ ВЕБ-КАМЕР: они часто отдают 0 или NaN (NaN не равен сам себе)
        self.fps = 30.0 if not fps or fps != fps else fps

        self.stopped = False
        # Ссылку на поток храним в объекте: иначе его невозможно дождаться в stop()
        self.thread = None
        # Очередь кадров (буфер). Помните о памяти: 128 кадров FullHD — это ~760 МБ
        self.Q = Queue(maxsize=queue_size)

    def start(self):
        # создаем поток чтения
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True # поток умрет вместе с главной программой
        self.thread.start()
        return self

    def update(self):
        # Бесконечный цикл чтения в потоке
        while True:
            if self.stopped:
                return

            # Производитель у очереди ровно один, поэтому проверка full() и следующий
            # за ней put() не могут "разъехаться"
            if not self.Q.full():
                ret, frame = self.stream.read()
                if not ret:
                    # Конец файла. Только поднимаем флаг и выходим: освобождать
                    # видеопоток здесь нельзя — главный поток еще разбирает буфер
                    # и может спросить метаданные
                    self.stopped = True
                    return
                # Добавляем кадр в очередь
                self.Q.put(frame)
            else:
                # Если очередь полна, ждем чуть-чуть
                time.sleep(0.01)

    def read(self):
        # Возвращаем кадр из очереди
        try:
            return self.Q.get_nowait()  # ← Не блокируется!
        except Empty:
            return None  # очередь пуста — вызывающий код обязан проверить результат

    def more(self):
        # Есть ли еще кадры?
        return not self.Q.empty()

    def stop(self):
        # Просим поток чтения остановиться
        self.stopped = True

        # И обязательно дожидаемся его. Поток может находиться внутри stream.read(),
        # а освобождать видеопоток, пока его читает кто-то другой, нельзя
        if self.thread is not None:
            self.thread.join(timeout=2.0)

        # Теперь источником никто не пользуется — освобождаем его
        self.stream.release()
