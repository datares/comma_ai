from datasets import *
from utils import *
import cv2
import os
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
from fastai.basic_data import DataBunch
from fastai.vision import *

if __name__ == "__main__":

#Steps: convert video to frames of pix: DONE! Not running again since files have already been saved.

    #frames = video_to_frames('train.mp4')
    #frames = enhance_frames('./train')

#loading data labels
    labels = load_data_labels('train.txt') #returns dataframe

#reading in frames into a dataset in python
    train = imageDataset(labels, './trainbright') #gives me a dataset that can be converted into dataloader
    complete_train = pd.DataFrame(columns = ['image', 'speed'])
    for i in range(5000): #too many open files, doing first 5000 frames as subset of whole dataset
       complete_train = complete_train.append({'image' : train.__getitem__(i)[0], 'speed' : train.__getitem__(i)[1]}, ignore_index=True)
    #print(complete_train) yay works

#splitting 'complete train (5000 rows)' into train and validation
    dataset_size = len(complete_train)
    validation_split = 0.3
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    dataload_train = DataLoader(complete_train, batch_size=1, shuffle=False, sampler=train_sampler, batch_sampler=None, num_workers=4)
    dataload_valid = DataLoader(complete_train, batch_size=1, shuffle=False, sampler=valid_sampler, batch_sampler=None, num_workers=4)

#Put dataloader into databunch
    databunch_for_model = DataBunch(dataload_train, dataload_valid, no_check=True) #no_check problem (sanity_check error)

#Run cnn_learner using the databunch we have
    learner = cnn_learner(databunch_for_model, models.resnet18)