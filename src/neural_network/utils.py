import asyncio
import threading
import queue
import io
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy import ndimage


class ProgressBridge:
    def __init__(self):
        self._bars: dict[str, tqdm] = {}
        self._lock = threading.Lock()
        self._q = queue.Queue()  # стандартный потокобезопасный Queue

    def add_bar(self, uid: str, total: int, **kwargs) -> tqdm:
        """Создаёт новый прогресс-бар (не пишет в консоль)"""
        bar = tqdm(total=total, file=io.StringIO(), **kwargs)
        with self._lock:
            self._bars[uid] = bar
        return bar

    def update(self, uid: str, n: int = 1) -> None:
        """Обновляет бар и сразу отправляет строку в очередь"""
        with self._lock:
            if uid in self._bars:
                self._bars[uid].update(n)
                # str(bar) даёт готовую строку вида " 40%|████      | 4/10 [00:00<00:00,  4.00it/s]"
                self._q.put_nowait((uid, str(self._bars[uid])))

    def close_bar(self, uid: str) -> None:
        """Закрывает бар и удаляет из памяти"""
        with self._lock:
            if uid in self._bars:
                self._bars[uid].close()
                del self._bars[uid]
        self._q.put_nowait((uid, None))  # None = сигнал завершения бара

    def mark_finished(self) -> None:
        """Сигнал о полном завершении вычислений"""
        self._q.put_nowait(("__DONE__", None))

    async def stream(self):
        """Асинхронный генератор: выдаёт (uid, bar_str) по мере обновления"""
        while True:
            # asyncio.to_thread безопасно блокирует только на ожидании item из Queue
            uid, msg = await asyncio.to_thread(self._q.get)
            if uid == "__DONE__":
                break
            yield uid, msg


def center_and_scale_digit(image: np.ndarray) -> np.ndarray:
    """Центрирование и масштабирование изображения (фон уже черный)"""
    
    image[image < 20] = 0  # Убираем шум
    img = Image.fromarray(image.astype('uint8').reshape(28, 28), mode='L')

    bbox = img.getbbox()
    if bbox == None:
        return image
    
    # Оригинальные длины сторон
    scaled_width = bbox[2] - bbox[0]
    scaled_height = bbox[3] - bbox[1]

    # Масштабированные
    if scaled_height > scaled_width:
        scaled_width = int(20.0 * scaled_width / scaled_height)
        scaled_height = 20
    else:
        scaled_height = int(20.0 * scaled_width / scaled_height)
        scaled_width = 20

    # Отмасштабированная картинка
    scaled_digit = img.crop(bbox).resize((scaled_width, scaled_height), Image.NEAREST)

    cmh, cmw = ndimage.center_of_mass(np.array(scaled_digit))

    # Стартовая точка рисования
    hstart = int(13.5 - cmh)
    wstart = int(13.5 - cmw)

    # Перенос на черный фон с центрированием
    centred_digit = Image.new('L', (28,28), 0)
    centred_digit.paste(scaled_digit, (wstart, hstart))

    # Конвертация в np.array
    return np.array(centred_digit).astype('float32').reshape(28, 28)