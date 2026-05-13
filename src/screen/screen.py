import asyncio
import threading
import time
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import ImageGrab
from ..neural_network.utils import ProgressBridge


class Screen(tk.Tk):
    def __init__(self, network):
        tk.Tk.__init__(self)
        self.title('Number recognition')
        self.network = network

        self.x = self.y = 0

        self.line_id = None
        self.line_points = []
        self.line_options = {}

        self.canvas = tk.Canvas(self, width=500, height=500, bg="white", cursor="cross")
        self.result_0 = tk.Label(self, text="...")
        self.result_1 = tk.Label(self, text="...")
        self.result_2 = tk.Label(self, text="...")
        self.result_3 = tk.Label(self, text="...")
        self.result_4 = tk.Label(self, text="...")
        self.result_5 = tk.Label(self, text="...")
        self.result_6 = tk.Label(self, text="...")
        self.result_7 = tk.Label(self, text="...")
        self.result_8 = tk.Label(self, text="...")
        self.result_9 = tk.Label(self, text="...")
        self.classify_button = tk.Button(self, text="Recognize", command=self.predict_number)
        self.clear_button = tk.Button(self, text="Clear", command=self.clear)
        self.fit_network_status = tk.Label(self, text="")

        self.epochs_input = tk.Spinbox(self, from_=1, to=100, textvariable=tk.IntVar(value=10))
        self.batch_size_input = tk.Spinbox(self, from_=1, to=5000, increment=16, textvariable=tk.IntVar(value=64))
        self.dataset_size_input = tk.Spinbox(self, from_=0, to=1, increment=0.1, textvariable=tk.DoubleVar(value=0.1))
        self.fit_network_button = tk.Button(self, text="Start fitting", command=self.start_fitting)
        self.stop_fitting_button = tk.Button(self, text="Stop fitting", command=self.stop_fitting)
        self.fit_network_log = tk.Text(self, state=DISABLED)

        # Сетка окна
        self.canvas.grid(row=0, rowspan=10, column=0, columnspan=2, pady=2, sticky=W)
        self.result_0.grid(row=0, column=2, pady=2, padx=2)
        self.result_1.grid(row=1, column=2, pady=2, padx=2)
        self.result_2.grid(row=2, column=2, pady=2, padx=2)
        self.result_3.grid(row=3, column=2, pady=2, padx=2)
        self.result_4.grid(row=4, column=2, pady=2, padx=2)
        self.result_5.grid(row=5, column=2, pady=2, padx=2)
        self.result_6.grid(row=6, column=2, pady=2, padx=2)
        self.result_7.grid(row=7, column=2, pady=2, padx=2)
        self.result_8.grid(row=8, column=2, pady=2, padx=2)
        self.result_9.grid(row=9, column=2, pady=2, padx=2)
        self.clear_button.grid(row=10, column=0, pady=2)
        self.classify_button.grid(row=10, column=1, columnspan=2, pady=2, padx=2)
        self.fit_network_status.grid(row=11, column=1, pady=2, padx=2)

        self.epochs_input.grid(row=0, column=3, pady=2, padx=2)
        self.batch_size_input.grid(row=1, column=3, pady=2, padx=2)
        self.dataset_size_input.grid(row=2, column=3, pady=2, padx=2)
        self.fit_network_button.grid(row=3, column=3, pady=2, padx=2)
        self.stop_fitting_button.grid(row=4, column=3, pady=2, padx=2)
        self.fit_network_log.grid(row=5, rowspan=7, column=3, pady=2, sticky=W)

        self.canvas.bind('<Button-1>', self.start_line)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.end_line)

        mainloop()

    def clear(self):
        self.canvas.delete("all")
        for n in range(10):
            getattr(self, 'result_' + str(n)).configure(text='...')

        self.fit_network_log.delete('1.0', tk.END)
        self.fit_network_status.configure(text='')

    def start_fitting(self):
        self.fitting_task = asyncio.run(self.fit_network())
    
    def stop_fitting(self):
        if self.fitting_task:
            self.fitting_task.cancel()
            self.classify_button.configure(state=NORMAL)
            self.fit_network_status.configure(text='Fitting stopped')

    async def fit_network(self):
        try:
            bridge = ProgressBridge()
            # Запускаем синхронные вычисления в отдельном потоке
            fit_thread = threading.Thread(target=self.network.fit, args=(bridge, int(self.epochs_input.get()), int(self.batch_size_input.get()), float(self.dataset_size_input.get())), daemon=True)
            show_thread = threading.Thread(target=lambda: asyncio.run(self.show_fit_progress(bridge)), daemon=True)

            fit_thread.start()
            show_thread.start()
        except Exception as e:
            self.fit_network_status.configure(text=str(e))
            self.classify_button.configure(state=NORMAL)

    async def show_fit_progress(self, bridge: ProgressBridge):
        self.fit_network_status.configure(text='Fitting...')
        self.fit_network_log.delete('1.0', tk.END)
        self.classify_button.configure(state=DISABLED)

        THROTTLE = 0.15  # обновляем UI не чаще 6-7 раз в секунду

        progress = ''
        latest_bar_str = ''
        last_ui_update = time.monotonic()

        # Отдаём строки в UI по мере готовности
        async for _, bar_str in bridge.stream():
            if bar_str is None:
                progress += str(latest_bar_str) + '\n'
            else:
                now = time.monotonic()
                if now - last_ui_update >= THROTTLE:
                    last_ui_update = now
                    self.fit_network_log.delete('1.0', tk.END)
                    self.fit_network_log.insert(progress + bar_str)
            latest_bar_str = bar_str
        else:
            self.fit_network_status.configure(text='Fitting done!')
            self.classify_button.configure(state=NORMAL)

    def predict_number(self):
        self.fit_network_status.configure(text='')

        try:
            x, y = (self.canvas.winfo_rootx(), self.canvas.winfo_rooty())
            width, height = (self.canvas.winfo_width(), self.canvas.winfo_height())

            img_coords = (x, y, x + width, y + height)
            img = ImageGrab.grab(img_coords)

            # изменение рзмера изобржений на 28x28
            img = img.resize((28, 28))

            # конвертируем rgb в grayscale
            img = img.convert('L')
            img = np.array(img).astype('float32')

            # img = img.reshape(28, 28, 1)
            # plt.imshow(img, cmap='gray')
            # plt.show()

            img = img.reshape(1, 28, 28, 1)

            # инвертируем чб цвета
            img = img * -1
            img = img + 255.0
            img = img / 255.0

            # предстказание цифры
            result = self.network.feedforward(img)
            print(result)

            predicted_number = np.argmax(result)

            for n in range(len(result)):
                getattr(self, 'result_' + str(n)).configure(text=str(n) + ', ' + str(int(result[n] * 100)) + '%' + (' - ✅' if predicted_number == n else ''), fg='green' if predicted_number == n else 'black')
        except:
            self.fit_network_status.configure(text='Error during recognition')

    def start_line(self, event):
        self.line_points.extend((event.x, event.y))

    def end_line(self, event=None):
        self.line_points.clear()
        self.line_id = None

    def draw_line(self, event):
        self.line_points.extend((event.x, event.y))
        if self.line_id is not None:
            self.canvas.delete(self.line_id)
        self.line_id = self.canvas.create_line(self.line_points, **self.line_options, width=42)
