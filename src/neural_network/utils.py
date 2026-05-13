import asyncio
import threading
import queue
from tqdm import tqdm
import io

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
