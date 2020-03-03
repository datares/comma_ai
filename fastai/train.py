from datasets import *
from utils import *
import cv2
import os
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
from fastai.basic_data import DataBunch
from fastai.vision import *
import re

if __name__ == "__main__":


    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]

    def sort_nicely(l):
        """ Sort the given list in the way that humans expect.
        """
        l.sort(key=alphanum_key)

    #frames = video_to_frames('train.mp4')

    path = './train'
    df = pd.DataFrame(columns=['name', 'label'])
    labels = load_data_labels('train.txt') #returns dataframe
    
    i = 0
    oslist = os.listdir('./train')
    sort_nicely(oslist)
    for img in oslist:
        df = df.append({'name': img, 'label': labels.label.values[i]}, ignore_index=True)
        i = i+1

    #creating databunch (splitting by random percent train/valid)
    data = (ImageList.from_df(path=path, df=df).split_by_rand_pct().label_from_df(cols=1,label_cls=FloatList).transform(get_transforms(), size=(224,224)).databunch().normalize(imagenet_stats))
    #create basic model on data
    learn = cnn_learner(data, models.resnet18)
    #for regression problem
    learn.loss = MSELossFlat
    #fit model
    learn.fit_one_cycle(1)












#Steps: convert video to frames of pix: DONE! Not running again since files have already been saved.

    #frames = video_to_frames('train.mp4')
    #frames = enhance_frames('./train')

#loading data labels
    #labels = load_data_labels('train.txt') #returns dataframe

#reading in frames into a dataset in python
    #train = imageDataset(labels, './trainbright') #gives me a dataset that can be converted into dataloader
    #complete_train = pd.DataFrame(columns = ['image', 'speed'])
    #for i in range(5000): #too many open files, doing first 5000 frames as subset of whole dataset
     #  complete_train = complete_train.append({'image' : train.__getitem__(i)[0], 'speed' : train.__getitem__(i)[1]}, ignore_index=True)
    #print(complete_train) yay works

#splitting 'complete train (5000 rows)' into train and validation
''' dataset_size = len(complete_train)
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
    learner = cnn_learner(databunch_for_model)
'''