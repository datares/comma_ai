import os 
import argparse
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from datasets import imageDataset
from utils import load_data_labels

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
	# transforms
	train_transform = transforms.Compose([transforms.ToTensor()])
	# TODO use transforms.Normalize w/ stdev and var
	# transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) <-- this is imagenet

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