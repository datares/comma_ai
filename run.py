import os 
import argparse
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset,DataLoader
from datasets import imageDataset
from utils import load_data_labels
import numpy as np

if not os.path.exists('./train/'):
	os.makedirs('./train/')
if not os.path.exists('./trainbright/'):
	os.makedirs('./trainbright')

if __name__ == "__main__":
	# command argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_path', type=str, default='.',
	help='Path to the data folder.')
	args = parser.parse_args()

	data_folder_path = args.data_path

	#calculate mean and stdev
	#
	'''
		adapted from 
		* https://medium.com/swlh/image-classification-tutorials-in-pytorch-transfer-learning-19ebc329e200
	'''
	mean = 0.
	std = 0.
	nb_samples = len(data)
	for data,_ in dataloader:
		batch_samples = data.size(0)
		data = data.view(batch_samples, data.size(1), -1)
		mean += data.mean(2).sum(0)
		std += data.std(2).sum(0)
	mean /= nb_samples
	std /= nb_samples

	# transforms
	# transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) <-- this is imagenet
	train_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([mean], [std])
	])

	# load the data labels into a pandas DataFrame
	ys = load_data_labels(data_folder_path + '/traintxt.txt')
	# initialize a training dataset
	trainset = imageDataset(ys, 'trainbright', transform =train_transform)

	# we don't shuffle yet because the order of the data matters
	# TODO we should probably shuffle, but pick batches of images that are related to themselves.
	train_loader = DataLoader(trainset, batch_size=8, shuffle=False, num_workers=4)

	# TODO can we do this with PyTorch/sklearn(!!!!)?
	# one example of how to create a validation set - sloppy, but this is one way to do it
	val = ys[16000:]
	val.reset_index(drop=True,inplace=True)