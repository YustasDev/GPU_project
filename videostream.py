from threading import Thread
import cv2
import time
from queue import Queue, Empty

class FileVideoStream:
    def __init__(self, path, queue_size=128):
        # Инициализация
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        # Очередь кадров (буфер)
        self.Q = Queue(maxsize=queue_size)

    def start(self):
        # создаем поток чтения
        t = Thread(target=self.update, args=())
        t.daemon = True # поток умрет вместе с главной программой
        t.start()
        return self

    def update(self):
        # Бесконечный цикл чтения в потоке
        while True:
            if self.stopped:
                return

            if not self.Q.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stop()
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
            return None

    def more(self):
        # Есть ли еще кадры?
        return not self.Q.empty()

    def stop(self):
        self.stopped = True
        self.stream.release()

        # Очистка очереди
        while not self.Q.empty():
            try:
                self.Q.get_nowait()
            except Empty:
                break