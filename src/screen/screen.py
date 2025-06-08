from tkinter import *
import tkinter as tk
import numpy as np
from PIL import ImageGrab, Image
from keras.models import load_model

class Screen(tk.Tk):
    def __init__(self, network):
        tk.Tk.__init__(self)

        self.__network = network

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
        self.classify_button = tk.Button(self, text="Распознать", command=self.predict_number)
        self.train_network_button = tk.Button(self, text="Переобучить", command=self.train_network)
        self.clear_button = tk.Button(self, text="Очистить", command=self.clear)
        self.train_network_status = tk.Label(self, text="")

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
        self.train_network_button.grid(row=10, column=1, pady=2)
        self.classify_button.grid(row=10, column=2, pady=2, padx=2)
        self.train_network_status.grid(row=11, column=1, pady=2, padx=2)

        self.canvas.bind('<Button-1>', self.start_line)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.end_line)

        mainloop()

    def clear(self):
        self.canvas.delete("all")

        for n in range(10):
            getattr(self, 'result_' + str(n)).configure(text='...')

    def train_network(self):
        if self.__network.train():
            self.train_network_status.configure(text='Обучено и сохранено')
        else:
            self.train_network_status.configure(text='Обучение не удалось')

    def predict_number(self):
        x, y = (self.canvas.winfo_rootx(), self.canvas.winfo_rooty())
        width, height = (self.canvas.winfo_width(), self.canvas.winfo_height())

        img_coords = (x, y, x + width, y + height)
        img = ImageGrab.grab(img_coords)

        # изменение рзмера изобржений на 28x28
        img = img.resize((28, 28))
        img.show()
        # конвертируем rgb в grayscale
        img = img.convert('L')
        img = np.array(img)
        # изменение размерности для поддержки модели ввода и нормализации; белый фон - 0, черная линия - 1
        img = img.reshape(1, 28, 28, 1)

        # нвертируем чб цвета
        img = img * -1
        img = img + 255.0
        img = img / 255.0
        # print(img[0])

        # предстказание цифры
        # res = self.__network.feedforward(img)
        # print(*res)
        # exit() #30eps лучший результат
        # todo:return np.argmax(res), max(res)
        model = load_model('mnist_30eps.h5')
        model.summary()
        result = model.predict(img)[0]
        print(*result)
        print(max(result))

        predicted_number = np.argmax(result)

        for n in range(len(result)):
            getattr(self, 'result_' + str(n)).configure(text=str(n) + ', ' + str(int(result[n] * 100)) + '%' + (' - ✅' if predicted_number == n else ''), fg='green' if predicted_number == n else 'black')

    def start_line(self, event):
        self.line_points.extend((event.x, event.y))

    def end_line(self, event=None):
        self.line_points.clear()
        self.line_id = None

    def draw_line(self, event):
        self.line_points.extend((event.x, event.y))
        if self.line_id is not None:
            self.canvas.delete(self.line_id)
        self.line_id = self.canvas.create_line(self.line_points, **self.line_options, width=24)#32
