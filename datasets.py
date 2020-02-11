## ANDREI REKESH - Jan 31
import os
from PIL import Image, ImageEnhance
import pandas as pd
import argparse
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

# define a custom PyTorch dataset that can be used to feed into a DataLoader
class imageDataset(Dataset):
	"""
	A custome PyTorch dataset that can be fed into a DataLoader.
	usage: imageDataset(dataframe: pandas.DataFrame, root_dir: str, transform: function)
	"""
	def __init__(self, dataframe, root_dir, transform=None):
		self.dataframe = dataframe
		self.root_dir = root_dir
		self.transform=transform

	def __len__(self):
		return len(self.dataframe)

	def __getitem__(self, idx):
		label = self.dataframe.label.values[idx]
		im = Image.open(self.root_dir+"/frame"+self.dataframe.frame.values[idx]+".jpg")
		#TODO: normalize image according to its own mean by channel
		if self.transform:
			im = self.transform(im)

		return im, label
