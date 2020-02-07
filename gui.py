import torch
import torch.nn.functional as F

from tkinter import *
from model.network import Network

USE_CUDA = torch.cuda.is_available()

class MainFrame(Frame):
    def __init__(self, master):
        super(MainFrame, self).__init__(master)
        pass

class App:
    def __init__(self, model):
        self.root = Tk()
        self.root.title('MNIST Visualizer')

        self.main_frame = MainFrame(self.root)

        self.model = Network(1, 128, 10, 10)
        self.model.load_state_dict(torch.load(model, map_location='cpu'))

        if USE_CUDA:
            self.model = self.model.cuda()

    def run(self):
        self.root.mainloop()
