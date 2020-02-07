import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from model.network import Network

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('image', help='Load image to classify')
	parser.add_argument('model', help='Load trained model')

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()

	net = Network(1, 64, 5, 10).to(device)
	net.load_state_dict(torch.load(args.model, map_location=device))

	image = Image.open(args.image).convert('L').resize((28, 28))
	image = ToTensor()(image).to(device)

	image = image.view(1, *image.shape)
	
	pred = F.softmax(net(image), dim=1)

	print('[Result]')
	print('Argmax:', pred.argmax(dim=1)[0].item())

	print('-----------------------')
	for i in range(10):
		print('{} : {:.4f}%'.format(i, pred[0][i].item() * 100))
