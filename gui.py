import torch
import torch.nn.functional as F

import tkinter as tk
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

from model.network import Network

USE_CUDA = torch.cuda.is_available()

class MainFrame(tk.Frame):
    def __init__(self, master):
        super(MainFrame, self).__init__(master)
        
        self.canvas = tk.Canvas(master, width=400, height=400, bg='black')
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        self.canvas.bind('<ButtonRelease-3>', self.clear_all)

        self.old_x, self.old_y = None, None

    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
            width=5, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)

        self.old_x, self.old_y = event.x, event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def clear_all(self, event):
        self.canvas.delete('all')

class App:
    def __init__(self, model):
        self.root = tk.Tk()
        self.root.title('MNIST Visualizer')

        self.main_frame = MainFrame(self.root)
        self.main_frame.pack(expand=tk.YES, fill='both')

        self.btn_run = tk.Button(self.root, text='RUN', command=self.btn_run_clicked)
        self.btn_run.pack(side=tk.BOTTOM)

        self.model = Network(1, 128, 10, 10)
        self.model.load_state_dict(torch.load(model, map_location='cpu'))

        if USE_CUDA:
            self.model = self.model.cuda()

    def btn_run_clicked(self):
        self.main_frame.canvas.postscript(file='temp.eps')

    def run(self):
        self.root.mainloop()
