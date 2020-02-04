## ANDREI REKESH - Jan 31
## Tony Xia - Feb 2 converted this into a function

import cv2
import os
from PIL import Image, ImageEnhance
import pandas as pd
import argparse

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

# loads the label data into a pandas.DataFrame object
def load_data_labels(filename: str) -> pd.DataFrame:
	# Load the labels into python with using a pandas DataFrame
	data = pd.read_csv(filename, sep=" ", header=None)
	# keep track of what frame corresponds to what speed
	data['frame'] = data.index
	# name the speed column "label"
	data.columns = ['label','frame']
	# swap the columns so that the dataframe reads, more conventionally, item name on the left and label on the right
	data = data[['frame','label']]
	# return the dataFrame object
	return data

def video_to_frames(video: str, destination='.') -> str:
	vidcap = cv2.VideoCapture(video)
	success,image = vidcap.read()
	count = 0

	# write raw frames to local files
	while success:
		cv2.imwrite(destination + '/train/frame%d.jpg' % count, image)     # save frame as JPEG file
		success,image = vidcap.read()
		print('Read a new frame: ', success)
		count += 1

def enhance_frames(train_path: str, destination_path='.'):
    #enhance the files, copy them into a different folder
    for img in os.listdir(train_path):
        im = Image.open('{}/{}'.format(train_path, img))

        brightness = ImageEnhance.Brightness(im)
        im = brightness.enhance(1.5)

        contrast = ImageEnhance.Contrast(im)
        im = contrast.enhance(2)

        sharpness = ImageEnhance.Sharpness(im)
        im = sharpness.enhance(2)

        im = im.save(destination + '/trainbright/%s' %img)


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


if __name__ == "__main__":

	# command argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_path', type=str, default='.',
		help='Path to the data folder.')
	args = parser.parse_args()
	data_folder_path = args.data_path


	# transform the image into something that can be fed to a neural network - we convert to tensor because neural networks want arrays of pixels
	train_transform = transforms.Compose([
	    transforms.ToTensor()
	    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) <-- this is imagenet
	])

	# load the data labels into a pandas DataFrame
	data_labels = load_data_labels(data_folder_path + '/traintxt.txt')

	#initialize a training dataset
	trainset = imageDataset(data_labels, data_folder_path + '/trainbright', transform=train_transform)

	# to test --> print(trainset.__getitem__(0))

	# create a dataloader - batch_size and num_workers will have to be tuned as time goes on

	# we don't shuffle yet because the order of the data matters
	# TODO we should probably shuffle, but pick batches of images that are related to themselves.
	# TODO how do we do this with pytorch?

	# TODO change batch_size
	# TODO change num_workers, your cpu has more cores for a reason!
	train_loader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1)

	# TODO can we do this with PyTorch/sklearn(!!!!)?
	# one example of how to create a validation set - sloppy, but this is one way to do it
	val = data_labels[16000:]
	val.reset_index(drop=True,inplace=True)
