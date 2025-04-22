from tkinter import *
import tkinter as tk

class Screen(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

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
        self.relearn_network_button = tk.Button(self, text="Переобучить", command=self.relearn_network)
        self.clear_button = tk.Button(self, text="Очистить", command=self.clear)

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
        self.relearn_network_button.grid(row=10, column=1, pady=2)
        self.classify_button.grid(row=10, column=2, pady=2, padx=2)

        self.canvas.bind('<Button-1>', self.start_line)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.end_line)

        mainloop()

    def clear(self):
        self.canvas.delete("all")

    def relearn_network(self):
        pass

    def predict_number(self):
        HWND = self.canvas.winfo_id()
        # rect = win32gui.GetWindowRect(HWND)  # получаем координату холста
        # im = ImageGrab.grab(rect)

        # digit, acc = predict_digit(im)
        # self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def start_line(self, event):
        self.line_points.extend((event.x, event.y))

    def end_line(self, event=None):
        self.line_points.clear()
        self.line_id = None

    def draw_line(self, event):
        self.line_points.extend((event.x, event.y))
        if self.line_id is not None:
            self.canvas.delete(self.line_id)
        self.line_id = self.canvas.create_line(self.line_points, **self.line_options, width=10)