import os
import sys
import re
import pandas as pd
import pathlib
from PIL import Image

def path_to_pd(img_dir_path):

	full_path = pathlib.Path(img_dir_path)
	print(full_path)

	files = full_path.rglob('*.jpg')

	folder_names = []
	file_paths = []
	file_names = []
	for fname in files:
		if fname.is_file():
			# print(fname.parts)
			file_paths.append(fname)
			file_names.append(fname.parts[-1])
			folder_names.append(fname.parts[-2])


	data_frame = {}
	data_frame.update({"breed_name": folder_names})
	data_frame.update({"file_path" : file_paths})
	data_frame.update({"file_name" : file_names})

	df = pd.DataFrame(data_frame)
	print(df)
	print(df.head())

	return df

if __name__ == '__main__':
	'''
	data utility functions used for processing data with in the data directory for dataloaders.
	'''

	STANFORD_DATA_DIR_PATH = r"C:\Users\Cameron\Documents\python projects\dog " \
							 "classification\data\stanford_dataset\images"
	TSINGHUA_DATA_DIR_PATH = r"C:\Users\Cameron\Documents\python projects\dog " \
							 r"classification\data\tsinghua_dataset\low-resolution"

	path_to_pd(TSINGHUA_DATA_DIR_PATH)