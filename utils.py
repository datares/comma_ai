import cv2
import pandas as pd
from pathlib import Path

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

# video is the path to the video
# destination is the final folder in which you want to store the train folder
# example: video_to_frames('train.mp4', './data')
# result: ./data/train/frame0.jpg ......
def video_to_frames(video: str, destination='.') -> str:
	vidcap = cv2.VideoCapture(video)
	success,image = vidcap.read()
	count = 0

	# make the directory to store the frames
	Path(destination + '/train').mkdir(parents=True, exist_ok=True)

	# write raw frames to local files
	while success:
		if not cv2.imwrite(destination + '/train/frame' + str(count) + '.jpg', image):    # save frame as JPEG file
			raise Exception("Could not write image")
		success,image = vidcap.read()
		print('Read a new frame: ', success)
		count += 1
def enhance_frames(train_path: str, destination='.'):
	# enhance the files, copy them into a different folder
	# creates the path to store transformed images
	Path(destination + '/trainbright').mkdir(parents=True, exist_ok=True)
	for img in os.listdir(train_path):
		im = Image.open('{}/{}'.format(train_path, img))

		brightness = ImageEnhance.Brightness(im)
		im = brightness.enhance(1.5)

		contrast = ImageEnhance.Contrast(im)
		im = contrast.enhance(2)

		sharpness = ImageEnhance.Sharpness(im)
		im = sharpness.enhance(2)

		im = im.save(destination + '/trainbright/%s' %img)

