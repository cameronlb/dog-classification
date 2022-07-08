import os
import sys
import re
import pathlib
import numpy as np
import pandas as pd
from PIL import Image

def path_to_pd_df(img_dir_path):

	full_path = pathlib.Path(img_dir_path)
	print("Converting data path into pandas data frame: \"{}\"".format(full_path))

	breed_match = re.compile("([a-zA-Z][a-zA-Z].*[a-zA-Z])")

	files = full_path.rglob('*.jpg')

	breed_names = []
	folder_names = []
	file_paths = []
	file_names = []
	file_ext = []
	# fname is absolute file path
	for fname in files:
		if fname.is_file():

			file_paths.append(fname)
			file_names.append(fname.parts[-1])
			folder_names.append(fname.parts[-2])
			file_ext.append(pathlib.Path(fname).suffix)

			match = re.search(breed_match, fname.parts[-2])
			if match:
				breed_names.append(match.group())


	data_frame = pd.DataFrame({'breed_name': breed_names,
							   'folder_name': folder_names,
							   'file_path' : file_paths,
							   'file_name' : file_names,
							   'file_ext': file_ext})

	unique_breed_names = data_frame['breed_name'].unique()
	# print("Unique breed names: {}".format(unique_breed_names))

	breed_num_dict = {}
	# create breed + unique number for class label
	for i, name in enumerate(unique_breed_names):
		# print(name, i)
		breed_num_dict.update({name: i + 1})

	# print(breed_num_dict)

	new_series = {"label": []}
	for breed_name in data_frame['breed_name']:

		if breed_num_dict[breed_name]:
			new_series["label"].append(breed_num_dict[breed_name])

	# print("new series: {}".format(new_series))
	new_series = pd.DataFrame(new_series)
	data_frame = pd.concat([new_series, data_frame], axis=1)

	return data_frame

if __name__ == '__main__':
	'''
	data utility functions used for processing data with in the data directory for dataloaders.
	'''

	STANFORD_DATA_DIR_PATH = r"C:\Users\Cameron\Documents\python projects\dog " \
							 "classification\data\stanford_dataset\images"
	TSINGHUA_DATA_DIR_PATH = r"C:\Users\Cameron\Documents\python projects\dog " \
							 r"classification\data\tsinghua_dataset\low-resolution"

	df_tsing = path_to_pd_df(TSINGHUA_DATA_DIR_PATH)
	df_stan = path_to_pd_df(STANFORD_DATA_DIR_PATH)

	print(df_stan.head())
	print(df_stan["breed_name"].unique())
