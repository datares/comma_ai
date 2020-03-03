import os 
import cv2
import argparse
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from datasets import imageDataset
from utils import load_data_labels, opticalFlowDense, save_image
from PIL import Image, ImageEnhance
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

if not os.path.exists('./train/'):
	os.makedirs('./train/')
if not os.path.exists('./trainbright/'):
	os.makedirs('./trainbright')
if not os.path.exists('./trainoptical/'):
	os.makedirs('./trainoptical')
def load(dataframe,root_dir,idx, transform):
	#label = dataframe.label.values[idx]
	im = mpimg.imread(f'{root_dir}/frame{dataframe.frame.values[idx]}.jpg')
	return im

if __name__ == "__main__":
	# command argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_path', type=str, default='.',
	help='Path to the data folder.')
	args = parser.parse_args()

	data_folder_path = args.data_path
	
	train_transform = transforms.Compose([transforms.ToTensor()])
	ys = load_data_labels(f'{data_folder_path}/train.txt')
	trainset = [load(ys, 'train', x, transform =train_transform) for x in range(len(ys))]
	path = 'trainoptical/test'
	for i in range(len(ys)):
		save_image(path, i,trainset)
	