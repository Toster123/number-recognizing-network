import asyncio
import threading
import queue
import io
import numpy as np
from typing import Optional
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

    def update(self, uid: str, n: int = 1, postfix: Optional[dict] = None) -> None:
        """Обновляет бар и сразу отправляет строку в очередь"""
        with self._lock:
            if uid in self._bars:
                self._bars[uid].n = n
                if postfix: 
                    self._bars[uid].set_postfix(postfix)

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


def to_string(s: object) -> str:
    return str(s)+"\n"

def accuracy(Y_pred: np.ndarray[np.float32], Y_true: np.ndarray[np.float32]) -> float:
    return np.mean(np.argmax(Y_pred, axis=1) == np.argmax(Y_true, axis=1))

def center_and_scale_digit(image: np.ndarray[np.uint8]) -> np.ndarray[np.float32]:
    """Центрирование и масштабирование изображения (фон уже черный)"""
    
    img = Image.fromarray(image.reshape(28, 28).astype('uint8'), mode='L')
    
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
    return np.array(centred_digit).reshape(1, 28, 28).astype('float32')

def im2col(X: np.ndarray[np.float32], kernel_size: tuple = (3, 3)) -> np.ndarray[np.float32]:
    """
    Извлекает патчи из входного тензора
    :param X: входной тензор (N, C, H, W)
    :param kernel_size: размер ядер (H, W)
    :return: X_col: (N*H_out*W_out, C*kH*kW)
    """
    N, C, H, W = X.shape
    H_out = H - kernel_size[0] + 1
    W_out = W - kernel_size[1] + 1
    
    # (N, C, H_out, W_out, kH, kW)
    windows = np.lib.stride_tricks.sliding_window_view(X, kernel_size, axis=(2, 3))
    
    # (N, H_out, W_out, C, kH, kW)
    windows = windows.transpose((0, 2, 3, 1, 4, 5))
    
    # Разворачиваем в 2D матрицу
    X_col = windows.reshape(N * H_out * W_out, C * kernel_size[0] * kernel_size[1])
    return X_col

def col2im(d_X_col: np.ndarray[np.float32], X_shape: tuple, kernel_size: tuple, output_size: tuple) -> np.ndarray[np.float32]:
    """
    Собирает градиенты по входу обратно в тензор (N, C, H, W)
    Так как патчи перекрываются, градиенты суммируются
    """
    N, C, _, _ = X_shape
    kH, kW = kernel_size
    _, H_out, W_out = output_size
    
    # Восстанавливаем форму (N, H_out, W_out, C, kH, kW)
    d_X_patch = d_X_col.reshape(N, H_out, W_out, C, kH, kW)

    # (N, C, H_out, W_out, kH, kW)
    d_X_patch = d_X_patch.transpose((0, 3, 1, 2, 4, 5))
    
    n_idx = np.arange(N)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    c_idx = np.arange(C)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

    # Индексы строк и столбцов для каждого положения окна k
    # (H_out, kH)
    h_idx = np.tile(np.arange(H_out).reshape((H_out, 1)), (1, kH)) + np.tile(np.arange(kH).reshape((1, kH)), (H_out, 1))
    # (W_out, kW)
    w_idx = np.tile(np.arange(W_out).reshape((W_out, 1)), (1, kW)) + np.tile(np.arange(kW).reshape((1, kW)), (W_out, 1))
    
    # Расширяем до d_X_patch.shape
    h_idx = h_idx[np.newaxis, np.newaxis, :, np.newaxis, :, np.newaxis] # (1, 1, H_out, 1, kH, 1)
    w_idx = w_idx[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, :] # (1, 1, 1, W_out, 1, kW)
    
    # Суммирование градиентов в перекрывающихся областях
    d_X = np.zeros(X_shape, dtype=d_X_patch.dtype)

    np.add.at(d_X, (n_idx, c_idx, h_idx, w_idx), d_X_patch)

    return d_X