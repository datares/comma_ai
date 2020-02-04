import cv2
import pandas as pd

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