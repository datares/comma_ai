from datasets import *
from utils import *
import cv2
import os
from PIL import Image, ImageEnhance


if __name__ == "__main__":

#Steps: convert video to frames of pix: DONE!

    #frames = video_to_frames('train.mp4')
    #frames = enhance_frames('./train')

#match pix with corresponding speed for each frame
    y_train = load_data_labels('train.txt')
    x_train = imageDataset('trainbright')
#reading in frames into a dataset in python

#reading in txt of actual speeds into file

#convert set of frames to dataloader

#Put dataloader into databunch

#Run cnn_learner using the databunch we have