from tkinter import *
import tkinter as tk

class Screen(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        self.canvas = tk.Canvas(self, width=600, height=600, bg="white", cursor="cross")
        self.result_1 = tk.Label(self, text="...")
        self.result = tk.Label(self, text="...")
        self.classify_button = tk.Button(self, text="Распознать", command=self.classify_handwriting)
        self.relearn_network_button = tk.Button(self, text="Переобучить", command=self.clear_all)
        self.clear_button = tk.Button(self, text="Очистить", command=self.clear_all)

        # Сетка окна
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.result.grid(row=0, column=1, pady=2, padx=2)
        self.classify_button.grid(row=1, column=1, pady=2, padx=2)
        self.clear_button.grid(row=1, column=0, pady=2)

        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

        mainloop()

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        # rect = win32gui.GetWindowRect(HWND)  # получаем координату холста
        # im = ImageGrab.grab(rect)

        # digit, acc = predict_digit(im)
        # self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')