import asyncio
import time
import threading
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import ImageGrab
from ..neural_network.utils import ProgressBridge, center_and_scale_digit


class Screen(tk.Tk):
    def __init__(self, network):
        tk.Tk.__init__(self)
        self.title('Number recognition')
        self.network = network

        self.line_id = None
        self.line_points = []
        self.line_options = {}

        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")

        self.grid_propagate(False)

        self.canvas = tk.Canvas(self, width=500, height=500, bg="white", cursor="cross")
        self.result_0 = tk.Label(self, text="...", width=40)
        self.result_1 = tk.Label(self, text="...", width=40)
        self.result_2 = tk.Label(self, text="...", width=40)
        self.result_3 = tk.Label(self, text="...", width=40)
        self.result_4 = tk.Label(self, text="...", width=40)
        self.result_5 = tk.Label(self, text="...", width=40)
        self.result_6 = tk.Label(self, text="...", width=40)
        self.result_7 = tk.Label(self, text="...", width=40)
        self.result_8 = tk.Label(self, text="...", width=40)
        self.result_9 = tk.Label(self, text="...", width=40)
        self.classify_button = tk.Button(self, text="Recognize", command=self.predict_number)
        self.clear_button = tk.Button(self, text="Clear", command=self.clear)
        self.fit_network_status = tk.Label(self, text="")

        self.ep_label = tk.Label(self, text="Epochs")
        self.batch_label = tk.Label(self, text="Batch")
        self.dataset_label = tk.Label(self, text="Dataset %")
        self.lr_label = tk.Label(self, text="LR")

        self.epochs_input = tk.Spinbox(self, from_=1, to=100, textvariable=tk.IntVar(value=5))
        self.batch_size_input = tk.Spinbox(self, from_=1, to=5000, increment=8, textvariable=tk.IntVar(value=16))
        self.dataset_size_input = tk.Spinbox(self, from_=0, to=1, increment=0.01, textvariable=tk.DoubleVar(value=0.3))
        self.learning_rate_input = tk.Spinbox(self, from_=0.001, to=0.5, increment=0.001, textvariable=tk.DoubleVar(value=0.07))
        self.fit_network_button = tk.Button(self, text="Start fitting", command=self.start_fitting)
        self.stop_fitting_button = tk.Button(self, text="Stop fitting", command=self.stop_fitting)
        self.fit_network_log = tk.Text(self)
        self.fit_task = None
        self.show_task = None
        self.fit_loop = None
        self.progress_loop = None

        # Сетка окна
        for i in range(10):
            self.rowconfigure(i, minsize=60, uniform="row_group")
        
        self.columnconfigure(2, weight=30, uniform="col_group")
        for i in range(3, 9):
            self.columnconfigure(i, weight=40, uniform="col_group")

        self.canvas.grid(row=0, rowspan=10, column=0, columnspan=2, pady=2, sticky=W)
        self.result_0.grid(row=0, column=2, pady=2, sticky=NSEW)
        self.result_1.grid(row=1, column=2, pady=2, sticky=NSEW)
        self.result_2.grid(row=2, column=2, pady=2, sticky=NSEW)
        self.result_3.grid(row=3, column=2, pady=2, sticky=NSEW)
        self.result_4.grid(row=4, column=2, pady=2, sticky=NSEW)
        self.result_5.grid(row=5, column=2, pady=2, sticky=NSEW)
        self.result_6.grid(row=6, column=2, pady=2, sticky=NSEW)
        self.result_7.grid(row=7, column=2, pady=2, sticky=NSEW)
        self.result_8.grid(row=8, column=2, pady=2, sticky=NSEW)
        self.result_9.grid(row=9, column=2, pady=2, sticky=NSEW)
        self.clear_button.grid(row=10, column=0, pady=2)
        self.classify_button.grid(row=10, column=1, pady=2, padx=2)

        self.ep_label.grid(row=0, column=3, pady=2, padx=2)
        self.batch_label.grid(row=0, column=4, pady=2, padx=2)
        self.dataset_label.grid(row=0, column=5, pady=2, padx=2)
        self.lr_label.grid(row=0, column=6, pady=2, padx=2)
        
        self.epochs_input.grid(row=1, column=3, pady=2, padx=2)
        self.batch_size_input.grid(row=1, column=4, pady=2, padx=2)
        self.dataset_size_input.grid(row=1, column=5, pady=2, padx=2)
        self.learning_rate_input.grid(row=1, column=6, pady=2, padx=2)
        self.fit_network_button.grid(row=1, column=7, pady=2, padx=2)
        self.stop_fitting_button.grid(row=1, column=8, pady=2, padx=2)
        self.fit_network_log.grid(row=2, rowspan=10, column=3, columnspan=6, pady=2, ipadx=110, ipady=110, sticky=W)
        self.fit_network_status.grid(row=12, column=3, columnspan=6, pady=2, padx=2)

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
    
    def stop_fitting(self):
        if self.progress_loop is not None and self.show_task is not None and not self.show_task.done():
            self.show_task.cancel()
            self.progress_loop.stop()
        if self.fit_loop is not None and self.fit_task is not None and not self.fit_task.done():
            self.fit_task.cancel()
            self.fit_loop.stop()

        self.fit_task = None
        self.show_task = None
        self.fit_loop = None
        self.progress_loop = None
        self.classify_button.configure(state=NORMAL)
        self.fit_network_status.configure(text='Fitting stopped. Weights file contains last epoch results.')

    def start_fitting(self):
        try:
            bridge = ProgressBridge()

            def start_loop(loop):
                asyncio.set_event_loop(loop)
                try:
                    loop.run_forever()
                finally:
                    loop.close()

            self.fit_loop = asyncio.new_event_loop()
            t1 = threading.Thread(target=start_loop, args=(self.fit_loop,), daemon=True)
            t1.start()

            self.progress_loop = asyncio.new_event_loop()
            t2 = threading.Thread(target=start_loop, args=(self.progress_loop,), daemon=True)
            t2.start()
            
            self.fit_task = asyncio.run_coroutine_threadsafe(self.network.fit(bridge, int(self.epochs_input.get()), int(self.batch_size_input.get()), float(self.dataset_size_input.get()), float(self.learning_rate_input.get())), self.fit_loop)
            self.show_task = asyncio.run_coroutine_threadsafe(self.show_fit_progress(bridge), self.progress_loop)
        except Exception as e:
            self.fit_network_status.configure(text=str(e))
            self.classify_button.configure(state=NORMAL)

    async def show_fit_progress(self, bridge: ProgressBridge):
        self.fit_network_status.configure(text='Weights backed up. Fitting...')
        self.fit_network_log.delete('1.0', tk.END)
        self.classify_button.configure(state=DISABLED)

        THROTTLE = 0.15  # обновляем UI не чаще 6-7 раз в секунду

        progress = ''
        latest_bar_str = ''
        last_ui_update = 0

        # Отдаём строки в UI по мере готовности
        async for _, bar_str in bridge.stream():
            await asyncio.sleep(0)

            if bar_str is None:
                progress += str(latest_bar_str) + '\n'
            else:
                now = time.monotonic()
                if now - last_ui_update >= THROTTLE:
                    last_ui_update = now
                    self.fit_network_log.delete('1.0', tk.END)
                    self.fit_network_log.insert('1.0', progress + bar_str)
            latest_bar_str = bar_str
        else:
            self.fit_network_log.delete('1.0', tk.END)
            self.fit_network_log.insert('1.0', progress)
            self.fit_network_status.configure(text='Fitting completed! See log for details. Weights file contains last epoch results.')
            self.classify_button.configure(state=NORMAL)

    def predict_number(self):
        self.fit_network_status.configure(text='')

        # try:
        x, y = (self.canvas.winfo_rootx(), self.canvas.winfo_rooty())
        width, height = (self.canvas.winfo_width(), self.canvas.winfo_height())

        img_coords = (x, y, x + width, y + height)
        img = ImageGrab.grab(img_coords)

        # изменение рзмера изобржений на 28x28
        img = img.resize((28, 28))

        # конвертируем rgb в grayscale
        img = img.convert('L')
        img = np.array(img).astype('uint8')

        # img = img.reshape(28, 28)
        # plt.imshow(img, cmap='gray')
        # plt.show()

        # инвертируем чб цвета
        img = img * -1
        img = img + 255

        # убираем шум
        img[img < 20] = 0

        # приведение к формату MNIST
        img = center_and_scale_digit(img).reshape(1, 1, 28, 28) / 255.0

        # предстказание цифры
        result = self.network(img)[0]
        print("Pred: ", result)

        predicted_number = np.argmax(result)

        for n in range(len(result)):
            getattr(self, 'result_' + str(n)).configure(text=str(n) + ', ' + str(int(result[n] * 100)) + '%' + (' - ✅' if predicted_number == n else ''), fg='green' if predicted_number == n else 'black')
        # except Exception as e:
        #     self.fit_network_status.configure(text='Error during recognition')
        #     self.fit_network_log.delete('1.0', tk.END)
        #     self.fit_network_log.insert('1.0', "Error during recognition: " + str(e))

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
