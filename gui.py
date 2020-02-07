import torch
import torch.nn.functional as F

from tkinter import *
from model.network import Network

USE_CUDA = torch.cuda.is_available()

class MainFrame(Frame):
    def __init__(self, master):
        super(MainFrame, self).__init__(master)
        
        self.canvas = Canvas(master, width=400, height=400, bg='black')
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        self.canvas.bind('<ButtonRelease-3>', self.clear_all)

        self.old_x, self.old_y = None, None

    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
            width=5, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=36)

        self.old_x, self.old_y = event.x, event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def clear_all(self, event):
        self.canvas.delete('all')

class App:
    def __init__(self, model):
        self.root = Tk()
        self.root.title('MNIST Visualizer')

        self.main_frame = MainFrame(self.root)

        self.message = Label(self.root, text='Press and Drag to draw')
        self.message.pack(side=BOTTOM)

        self.model = Network(1, 128, 10, 10)
        self.model.load_state_dict(torch.load(model, map_location='cpu'))

        if USE_CUDA:
            self.model = self.model.cuda()

    def run(self):
        self.root.mainloop()
