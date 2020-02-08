import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage

from model.network import Network

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('image', help='Load image to classify')
	parser.add_argument('model', help='Load trained model')

	return parser.parse_args()

def to_img(tensor: torch.Tensor):
	return ToPILImage()(tensor.cpu().view(1, 28, 28))

if __name__ == '__main__':
	args = parse_args()

	net = Network(1, 64, 5, 10).to(device)
	net.load_state_dict(torch.load(args.model, map_location=device))

	image = Image.open(args.image).convert('L').resize((28, 28))
	image = ToTensor()(image).to(device)

	image = image.view(1, *image.shape)
	
	pred, feat = net.predict_with_feature(image)
	pred = F.softmax(pred, dim=1)

	grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.2)

	orig_img_plot = plt.subplot(grid[0, 0])
	orig_img_plot.set_title('input image')
	orig_img_plot.xaxis.set_major_locator(plt.NullLocator())
	orig_img_plot.yaxis.set_major_locator(plt.NullLocator())
	orig_img_plot.imshow(to_img(image), cmap='gray')

	feat_map_plot = plt.subplot(grid[0, 1])
	feat_map_plot.set_title('feature map')
	feat_map_plot.xaxis.set_major_locator(plt.NullLocator())
	feat_map_plot.yaxis.set_major_locator(plt.NullLocator())
	feat_map_plot.imshow(to_img(feat), cmap='gray')

	last_plot = plt.subplot(grid[1:, 0:])
	last_plot.set_xlabel('Prob')
	last_plot.set_ylabel('Number')
	last_plot.set_xlim((0, 100))
	last_plot.set_yticks(np.arange(10))
	last_plot.barh(np.arange(10), (100. * pred).cpu().numpy()[0])

	plt.show()
